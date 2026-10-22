"""Scale-Gated Pyramid Pooling Module (SG-PPM).

Enhances the standard PPM by prepending a lightweight scale gate that
adaptively weights each pooling branch before concatenation.  The gate
uses global average pooling on C4 followed by a two-layer MLP with
softmax normalisation, so branches whose context scale is more relevant
to the current input receive higher weight.

Drop-in replacement for ``PPM + bottleneck`` in ``UPerHead.psp_forward``:
a single forward call produces the same ``(B, channels, H, W)`` output.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGPPM(nn.Module):
    """Scale-Gated Pyramid Pooling Module.

    Args:
        in_channels: Channel width of the input feature (e.g. C4 = 768).
        channels: Output channel width after fusion (e.g. 512).
        pool_scales: Pooling output sizes for each PPM branch.
        align_corners: ``align_corners`` for bilinear upsampling.
        reduction: Channel reduction ratio for the scale gate MLP.
    """

    def __init__(
        self,
        in_channels: int = 768,
        channels: int = 512,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        align_corners: bool = False,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.pool_scales = tuple(pool_scales)
        self.align_corners = align_corners
        num_scales = len(self.pool_scales)

        # --- PPM multi-scale branches ---
        self.ppm_branches = nn.ModuleList()
        for scale in self.pool_scales:
            self.ppm_branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # --- Scale gate: GAP -> MLP -> softmax per-scale weights ---
        hidden = max(in_channels // reduction, 16)
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_scales, kernel_size=1, bias=True),
        )

        # --- Bottleneck: raw C4 + weighted branches -> channels ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + num_scales * channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # per-scale weights: (B, num_scales, 1, 1)
        scale_weights = torch.softmax(self.scale_gate(x), dim=1)

        ppm_outs = [x]
        for i, branch in enumerate(self.ppm_branches):
            pooled = branch(x)
            pooled = F.interpolate(
                pooled, size=(H, W),
                mode="bilinear", align_corners=self.align_corners,
            )
            pooled = pooled * scale_weights[:, i : i + 1, :, :]
            ppm_outs.append(pooled)

        return self.bottleneck(torch.cat(ppm_outs, dim=1))


__all__ = ["SGPPM"]


if __name__ == "__main__":
    x = torch.randn(2, 768, 16, 16)
    module = SGPPM(in_channels=768, channels=512)
    y = module(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    assert y.shape == (2, 512, 16, 16), f"Expected (2,512,16,16), got {y.shape}"

    total = sum(p.numel() for p in module.parameters())
    print(f"Params: {total / 1e3:.1f}K")

    y.sum().backward()
    print("Backward OK")
