import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        padding=1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.SyncBatchNorm(out_channels),
        ]

        if act:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DiagonalStripConv(nn.Module):
    """Depthwise KxK conv whose effective support is a single diagonal line.

    A standard ``Conv2d`` cannot directly express "convolution along the main
    or anti diagonal only".  We instead allocate a depthwise KxK kernel and
    multiply its weight by a fixed 0/1 mask in :meth:`forward`, so only the
    parameters at diagonal positions ever contribute to the output.  The mask
    is registered as a buffer (no gradient) and the underlying ``Conv2d``
    weight is left untouched between forward passes — gradients flow only into
    the diagonal entries because off-diagonal positions are multiplied by 0.

    Args:
        channels: number of input/output channels (depthwise).
        kernel_size: odd KxK window size.
        direction: ``'main'`` for top-left to bottom-right (``i == j``) or
            ``'anti'`` for top-right to bottom-left (``i + j == K - 1``).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        direction: str = "main",
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if direction not in ("main", "anti"):
            raise ValueError("direction must be 'main' or 'anti'")

        self.kernel_size = kernel_size
        self.direction = direction
        self.padding = kernel_size // 2

        # Depthwise KxK conv; off-diagonal weights will be masked to zero.
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=channels,
            bias=False,
        )

        mask = torch.zeros(1, 1, kernel_size, kernel_size)
        for i in range(kernel_size):
            j = i if direction == "main" else (kernel_size - 1 - i)
            mask[0, 0, i, j] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.conv.weight * self.mask
        return F.conv2d(
            x,
            masked_weight,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=self.conv.groups,
        )


class StableStripSpatialAttention(nn.Module):
    """Stable strip spatial attention for high-level SAR features.

    The default behaviour stays identical to the original implementation:
    a depthwise local-context conv followed by horizontal and vertical strip
    convs, then a 1x1 projection and sigmoid gating.  When
    ``use_diagonal_strips`` is ``True``, two additional diagonal strip convs
    (main + anti) are applied to the original input ``x`` and **added** to the
    horizontal/vertical strip features before the projection — extending the
    attention support from a cross to an asterisk pattern, which is more
    aligned with oil-spill's elongated yet often oblique structures.
    """

    def __init__(
        self,
        channels: int,
        strip_kernel_size: int = 7,
        use_diagonal_strips: bool = False,
    ) -> None:
        super().__init__()

        if strip_kernel_size % 2 == 0:
            raise ValueError("strip_kernel_size must be odd")

        self.use_diagonal_strips = use_diagonal_strips

        self.local_context_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False,
        )
        self.horizontal_strip_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, strip_kernel_size),
            padding=(0, strip_kernel_size // 2),
            groups=channels,
            bias=False,
        )
        self.vertical_strip_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(strip_kernel_size, 1),
            padding=(strip_kernel_size // 2, 0),
            groups=channels,
            bias=False,
        )

        if use_diagonal_strips:
            # Diagonals operate directly on x (parallel to the H/V branch),
            # mirroring how strip context attends along an axis.
            self.main_diag_conv = DiagonalStripConv(
                channels=channels,
                kernel_size=strip_kernel_size,
                direction="main",
            )
            self.anti_diag_conv = DiagonalStripConv(
                channels=channels,
                kernel_size=strip_kernel_size,
                direction="anti",
            )

        self.attention_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.gamma = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.local_context_conv(x)
        attn = self.horizontal_strip_conv(attn)
        attn = self.vertical_strip_conv(attn)
        if self.use_diagonal_strips:
            attn = attn + self.main_diag_conv(x) + self.anti_diag_conv(x)
        attn = self.attention_projection(attn)
        attn = torch.sigmoid(attn)
        return x + self.gamma * (x * attn)


class StripContextBranch(nn.Module):
    """Strip context branch for elongated SAR oil-spill structures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strip_kernel_size: int = 7,
        use_diagonal_strips: bool = False,
    ) -> None:
        super().__init__()

        self.pre_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.SyncBatchNorm(in_channels),
            nn.GELU(),
        )
        self.strip_attn = StableStripSpatialAttention(
            channels=in_channels,
            strip_kernel_size=strip_kernel_size,
            use_diagonal_strips=use_diagonal_strips,
        )
        self.out_proj = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_proj(x)
        x = self.strip_attn(x)
        return self.out_proj(x)


class SPCM(nn.Module):
    """Strip-aware Pyramid Context Module.

    Designed as a standalone candidate for replacing UPerHead's original PPM
    later. It enhances C4 high-level features with pyramid pooling, strip
    context, optional local context, and a 1x1 projection of the raw input
    (aligned with PPM's ``psp_outs = [x]``) without changing the current model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        pool_scales=(1, 2, 3, 6),
        strip_kernel_size: int = 7,
        use_local_branch: bool = True,
        use_diagonal_strips: bool = False,
    ) -> None:
        super().__init__()

        self.pool_scales = pool_scales
        self.use_local_branch = use_local_branch
        self.use_diagonal_strips = use_diagonal_strips

        self.pool_branches = nn.ModuleList()
        for scale in pool_scales:
            self.pool_branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((scale, scale)),
                    ConvBNAct(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        padding=0,
                    ),
                )
            )

        self.strip_branch = StripContextBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            strip_kernel_size=strip_kernel_size,
            use_diagonal_strips=use_diagonal_strips,
        )

        if use_local_branch:
            self.local_branch = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )

        self.identity_branch = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

        num_branches = len(pool_scales) + 2  # strip + identity
        if use_local_branch:
            num_branches += 1

        self.fusion = ConvBNAct(
            out_channels * num_branches,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        outs = []

        for branch in self.pool_branches:
            pooled = branch(x)
            pooled = F.interpolate(
                pooled,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            outs.append(pooled)

        outs.append(self.strip_branch(x))

        if self.use_local_branch:
            outs.append(self.local_branch(x))

        outs.append(self.identity_branch(x))

        out = torch.cat(outs, dim=1)
        return self.fusion(out)


if __name__ == "__main__":
    x = torch.randn(2, 768, 16, 16)
    for diag in (False, True):
        module = SPCM(
            in_channels=768,
            out_channels=256,
            pool_scales=(1, 2, 3, 6),
            strip_kernel_size=7,
            use_local_branch=True,
            use_diagonal_strips=diag,
        )
        y = module(x)
        print(f"diag={diag} Input={tuple(x.shape)} Output={tuple(y.shape)}")
