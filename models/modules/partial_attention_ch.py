"""Partial Channel Attention module.

Splits the input channels into two groups:
- A small fraction (1/split_factor) passes through a 3x3 convolution
  for local spatial mixing.
- The remaining channels are re-weighted by a lightweight channel
  attention mechanism based on per-channel mean and std statistics.

The two groups are concatenated back, preserving the original channel
count and spatial resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    """Per-channel attention via mean/std statistics.

    Computes the spatial mean and standard deviation for each channel,
    concatenates them along the last dimension ``(B, C, 1, 2)``, and
    passes through a ``(1, 2)`` convolution followed by Hardsigmoid to
    produce a ``(B, C, 1, 1)`` channel-wise scaling factor.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 2), bias=False)
        self.act = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        flat = x.reshape(b, c, -1)
        mean = flat.mean(dim=-1).view(b, c, 1, 1)
        std = flat.std(dim=-1).view(b, c, 1, 1)
        combined = torch.cat([mean, std], dim=-1)       # (B, C, 1, 2)
        attn = self.act(self.conv(combined))             # (B, C, 1, 1)
        return x * attn


class PartialAttentionCH(nn.Module):
    """Partial-channel attention: local conv + channel attention in parallel.

    Args:
        channels: Total number of input/output channels.
        split_factor: Denominator for the conv-branch fraction.
            ``channels // split_factor`` channels go through 3x3 conv;
            the rest go through ``ChannelAttentionModule`` + SyncBN.
    """

    def __init__(self, channels: int, split_factor: int = 4) -> None:
        super().__init__()
        if channels % split_factor != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by "
                f"split_factor ({split_factor})"
            )

        self.conv_channels = channels // split_factor
        self.attn_channels = channels - self.conv_channels

        self.partial_conv = nn.Conv2d(
            self.conv_channels, self.conv_channels, 3, padding=1, bias=False,
        )
        self.channel_attention = ChannelAttentionModule(self.attn_channels)
        self.bn = nn.SyncBatchNorm(self.attn_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_part, attn_part = x.split(
            [self.conv_channels, self.attn_channels], dim=1,
        )
        conv_part = self.partial_conv(conv_part)
        attn_part = self.bn(self.channel_attention(attn_part))
        return torch.cat([conv_part, attn_part], dim=1)


if __name__ == "__main__":
    x = torch.randn(2, 512, 64, 64)
    m = PartialAttentionCH(channels=512, split_factor=4)
    y = m(x)
    print("in:", x.shape, "out:", y.shape)
    assert x.shape == y.shape
    total = sum(p.numel() for p in m.parameters())
    print(f"Params: {total / 1e3:.1f}K")
