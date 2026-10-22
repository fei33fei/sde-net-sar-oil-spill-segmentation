"""Gated Top-Down Fusion for FPN-style decoders.

Replaces the unconditional element-wise addition in the standard FPN
top-down pathway with a learned sigmoid gate that controls how much of
each upsampled deeper feature is added to the shallower lateral:

    deep_up  = upsample(lat[i])
    gate     = sigmoid(Conv([lat[i-1], deep_up]))
    lat[i-1] = lat[i-1] + gate * deep_up

This allows the network to suppress noisy shallow-level contributions
(common in SAR imagery) while preserving useful spatial detail.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedTopDownFusion(nn.Module):
    """Learned gate for each FPN top-down connection.

    Args:
        channels: Unified channel width of every lateral.
        num_levels: Number of pyramid levels (typically 4).
    """

    def __init__(self, channels: int, num_levels: int = 4) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be >= 2")
        self.num_levels = num_levels
        self.gate_convs = nn.ModuleList(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
            for _ in range(num_levels - 1)
        )

    def forward(self, lat: list[torch.Tensor]) -> None:
        """Apply gated top-down fusion **in-place** on *lat*."""
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
