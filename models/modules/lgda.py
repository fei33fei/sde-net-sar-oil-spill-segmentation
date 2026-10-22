import torch
import torch.nn as nn


class LGDA(nn.Module):
    """Lightweight Gated Denoising Attention.

    A standalone lightweight attention module for shallow SAR features. It is
    intended for later use after UPerHead C1/C2/C3 lateral convolutions, but it
    is not wired into the existing model by default.

    ``alpha`` is initialized to ``1e-3`` for a mild warm-start on the residual
    path; ``beta`` is ``1.0`` so the low-information branch contributes from
    the start while the overall update remains small due to ``alpha``.
    """

    def __init__(
        self,
        channels: int,
        num_groups: int = 16,
        spatial_kernel_size: int = 7,
        channel_kernel_size: int = 5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if spatial_kernel_size % 2 == 0:
            raise ValueError("spatial_kernel_size must be odd")
        if channel_kernel_size % 2 == 0:
            raise ValueError("channel_kernel_size must be odd")

        num_groups = min(num_groups, channels)

        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=channels,
            eps=eps,
            affine=True,
        )

        self.spatial_gate = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=True,
        )

        self.channel_gate = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=channel_kernel_size,
            padding=channel_kernel_size // 2,
            bias=False,
        )

        self.low_filter = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.SyncBatchNorm(channels),
        )

        self.beta = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.full((1,), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_feat = self.norm(x)

        avg_map = norm_feat.mean(dim=1, keepdim=True)
        max_map = norm_feat.amax(dim=1, keepdim=True)
        spatial = torch.sigmoid(
            self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        )

        gap = norm_feat.mean(dim=(2, 3))
        channel = torch.sigmoid(
            self.channel_gate(gap.unsqueeze(1)).squeeze(1)
        ).unsqueeze(-1).unsqueeze(-1)

        gate = spatial * channel
        high_info = x * gate
        low_info = x * (1.0 - gate)
        low_filtered = self.low_filter(low_info)

        clean = high_info + self.beta * low_filtered
        return x + self.alpha * (clean - x)


if __name__ == "__main__":
    x = torch.randn(2, 256, 64, 64)
    model = LGDA(channels=256, num_groups=16)
    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
