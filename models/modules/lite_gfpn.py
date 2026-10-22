"""Lightweight Gated FPN (LiteGFPN).

A streamlined multi-scale fusion module that retains the three core
geometric enhancements from ``ClassAwareGatedFPN`` while dropping all
class-prototype machinery:

1. **GFM** -- Global Feature Modulation: the deepest lateral's GAP vector
   modulates all shallower laterals before fusion.
2. **Gated top-down** -- learned sigmoid gates control how much of each
   upsampled deeper feature is added to the shallower lateral, replacing
   the vanilla unconditional addition.
3. **Learnable scale weights** -- a ``softmax``-normalised scalar per
   pyramid level weights each FPN output before concatenation+fusion.

Input contract: a list of ``num_levels`` feature maps that **already share
the same channel width** (e.g. after external lateral 1x1 projections and
PPM bottleneck on the deepest stage).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        ),
        nn.SyncBatchNorm(out_channels),
        nn.ReLU(inplace=True),
    )


class LiteGFPN(nn.Module):
    """GFM + gated top-down + learnable scale-weighted fusion.

    Args:
        channels: Unified channel width of every input lateral.
        num_levels: Number of pyramid levels (typically 4).
    """

    def __init__(self, channels: int, num_levels: int = 4) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be >= 2")

        self.channels = channels
        self.num_levels = num_levels

        # GFM: deepest GAP → 1x1 → sigmoid → modulate shallower laterals
        self.gfm_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gfm_gamma = nn.Parameter(torch.tensor(1e-3))

        # Gated top-down: per-level gate conv
        self.gate_convs = nn.ModuleList(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
            for _ in range(num_levels - 1)
        )

        # Per-level 3x3 refinement (applied to all but deepest)
        self.fpn_convs = nn.ModuleList(
            _conv_bn_relu(channels, channels, 3, 1)
            for _ in range(num_levels - 1)
        )

        # Learnable scale weights (softmax-normalised before use)
        self.scale_weights = nn.Parameter(torch.zeros(num_levels))

        # Final concat → 3x3 fusion
        self.fusion = _conv_bn_relu(
            num_levels * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def _apply_gfm(self, lat: list[torch.Tensor]) -> None:
        top = lat[-1]
        gfm_vec = F.adaptive_avg_pool2d(top, output_size=1)
        gfm_attn = torch.sigmoid(self.gfm_proj(gfm_vec))
        g = self.gfm_gamma
        for i in range(self.num_levels - 1):
            lat[i] = lat[i] * (1.0 + g * gfm_attn)

    def _gated_top_down(self, lat: list[torch.Tensor]) -> None:
        for i in range(self.num_levels - 1, 0, -1):
            deep_up = F.interpolate(
                lat[i],
                size=lat[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            gate_in = torch.cat([lat[i - 1], deep_up], dim=1)
            gate = torch.sigmoid(self.gate_convs[i - 1](gate_in))
            lat[i - 1] = lat[i - 1] + gate * deep_up

    def forward(self, laterals: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            laterals: ``num_levels`` tensors, each ``(B, channels, H_i, W_i)``,
                ordered from finest to coarsest resolution.

        Returns:
            Fused feature map ``(B, channels, H_0, W_0)`` at the finest
            resolution.
        """
        if len(laterals) != self.num_levels:
            raise ValueError(
                f"expected {self.num_levels} laterals, got {len(laterals)}"
            )

        lat = list(laterals)

        self._apply_gfm(lat)
        self._gated_top_down(lat)

        fpn_outs: list[torch.Tensor] = []
        for i in range(self.num_levels - 1):
            fpn_outs.append(self.fpn_convs[i](lat[i]))
        fpn_outs.append(lat[-1])

        target_hw = lat[0].shape[2:]
        for i in range(1, self.num_levels):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        w = F.softmax(self.scale_weights, dim=0)
        weighted = [w[i] * fpn_outs[i] for i in range(self.num_levels)]
        feats = torch.cat(weighted, dim=1)
        return self.fusion(feats)


if __name__ == "__main__":
    laterals = [
        torch.randn(2, 512, 128, 128),
        torch.randn(2, 512, 64, 64),
        torch.randn(2, 512, 32, 32),
        torch.randn(2, 512, 16, 16),
    ]
    m = LiteGFPN(channels=512, num_levels=4)
    y = m(laterals)
    print("out:", y.shape)
    total = sum(p.numel() for p in m.parameters())
    print(f"Params: {total / 1e6:.2f}M")
