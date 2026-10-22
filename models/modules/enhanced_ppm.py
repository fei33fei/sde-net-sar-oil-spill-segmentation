"""Enhanced Pyramid Pooling Module with directional context branches.

Fuses three families of context extraction on the deepest backbone stage (C4):

1. **Identity** -- 1x1 projection of the raw input.
2. **Classic PPM** -- adaptive average pooling at multiple scales (default 1/2/3/6),
   bilinear upsample, then 1x1 projection (same as vanilla PPM).
3. **Directional pooling** -- horizontal strip pooling, vertical strip pooling,
   main-diagonal and anti-diagonal strip convolutions (reuses
   ``DiagonalStripConv`` from ``spcm.py``).

Every branch is followed by a lightweight depthwise 3x3 refinement before
concatenation and a 3x3 fusion convolution that maps to ``out_channels``.

Drop-in replacement for ``PPM + bottleneck`` in ``UPerHead.psp_forward``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.spcm import DiagonalStripConv


class _ConvBNReLU(nn.Module):
    """Conv2d + SyncBatchNorm + ReLU helper."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, groups=groups, bias=False),
            nn.SyncBatchNorm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _LightDWRefine(nn.Module):
    """Depthwise 3x3 + SyncBN + ReLU refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.SyncBatchNorm(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.dw(x)))


class EnhancedPPM(nn.Module):
    """Identity + PPM(4) + Directional(4) = 9-branch pyramid context.

    Args:
        in_channels: Channel width of the input feature map (e.g. C4=768).
        out_channels: Channel width of the fused output (e.g. 512).
        branch_channels: Per-branch intermediate width before concat.
            Defaults to ``out_channels`` when *None*.
        pool_scales: Scales for the classic PPM branches.
        diag_kernel_size: Odd kernel size for ``DiagonalStripConv``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        branch_channels: int | None = None,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        diag_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        bc = branch_channels if branch_channels is not None else out_channels
        self.bc = bc

        # --- 1) Identity branch ---
        self.identity = _ConvBNReLU(in_channels, bc, kernel_size=1)

        # --- 2) Classic PPM branches ---
        self.pool_branches = nn.ModuleList()
        for s in pool_scales:
            self.pool_branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    _ConvBNReLU(in_channels, bc, kernel_size=1),
                )
            )

        # --- 3) Directional branches ---
        # 3a) Horizontal strip pooling  (1, W)
        self.h_proj = _ConvBNReLU(in_channels, bc, kernel_size=1)
        # 3b) Vertical strip pooling    (H, 1)
        self.v_proj = _ConvBNReLU(in_channels, bc, kernel_size=1)
        # 3c) Main-diagonal strip conv
        self.diag_main_conv = DiagonalStripConv(in_channels, diag_kernel_size, direction="main")
        self.diag_main_proj = _ConvBNReLU(in_channels, bc, kernel_size=1)
        # 3d) Anti-diagonal strip conv
        self.diag_anti_conv = DiagonalStripConv(in_channels, diag_kernel_size, direction="anti")
        self.diag_anti_proj = _ConvBNReLU(in_channels, bc, kernel_size=1)

        # --- 4) Per-branch light DW refine ---
        num_branches = 1 + len(pool_scales) + 4
        self.refines = nn.ModuleList(_LightDWRefine(bc) for _ in range(num_branches))

        # --- 5) Fusion ---
        self.fusion = nn.Sequential(
            nn.Conv2d(num_branches * bc, out_channels, 3, padding=1, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        outs: list[torch.Tensor] = []

        # 1) Identity
        outs.append(self.identity(x))

        # 2) PPM
        for branch in self.pool_branches:
            pooled = branch(x)
            pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(pooled)

        # 3a) Horizontal strip: pool to (1, W), broadcast back to (H, W)
        h_pool = F.adaptive_avg_pool2d(x, (1, w))
        outs.append(self.h_proj(h_pool.expand_as(x)))

        # 3b) Vertical strip: pool to (H, 1), broadcast back to (H, W)
        v_pool = F.adaptive_avg_pool2d(x, (h, 1))
        outs.append(self.v_proj(v_pool.expand_as(x)))

        # 3c) Main diagonal
        outs.append(self.diag_main_proj(self.diag_main_conv(x)))

        # 3d) Anti diagonal
        outs.append(self.diag_anti_proj(self.diag_anti_conv(x)))

        # 4) Per-branch DW refine
        outs = [refine(o) for refine, o in zip(self.refines, outs)]

        # 5) Concat + fusion
        return self.fusion(torch.cat(outs, dim=1))


__all__ = ["EnhancedPPM"]


if __name__ == "__main__":
    x = torch.randn(2, 768, 16, 16)
    m = EnhancedPPM(in_channels=768, out_channels=512)
    y = m(x)
    print(f"Input: {tuple(x.shape)}  Output: {tuple(y.shape)}")
