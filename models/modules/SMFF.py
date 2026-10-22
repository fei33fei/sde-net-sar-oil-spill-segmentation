"""Structural-aware multi-scale fusion (SMFF / SMMM) for paired feature maps.

Structured skip fusion with multi-scale extraction and channel masking; suitable
for encoder/decoder paths that share spatial resolution. See arXiv:2512.05494.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_padding(
    kernel_size: int | tuple[int, int],
    padding: Optional[int | tuple[int, int]] = None,
    dilation: int = 1,
) -> int | tuple[int, int]:
    """Compute padding so convolution preserves spatial size (odd kernels)."""
    if dilation > 1:
        if isinstance(kernel_size, int):
            effective_k = dilation * (kernel_size - 1) + 1
        else:
            effective_k = tuple(dilation * (k - 1) + 1 for k in kernel_size)
        kernel_size = effective_k
    if padding is not None:
        return padding
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(k // 2 for k in kernel_size)


def _group_norm_groups(num_channels: int, preferred: int = 4) -> int:
    """Pick a valid GroupNorm group count (divides ``num_channels``)."""
    g = min(preferred, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return max(1, g)


class _BasicConv(nn.Module):
    """Conv2d + SyncBatchNorm + activation (project style, DDP-friendly)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        pad = _same_padding(kernel_size, padding, dilation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.SyncBatchNorm(out_channels)
        if activation is True:
            self.act: nn.Module = nn.SiLU(inplace=True)
        elif isinstance(activation, nn.Module):
            self.act = activation
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DepthwiseConv(_BasicConv):
    """Depthwise separable-style conv (groups = gcd(in, out))."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,
    ) -> None:
        g = math.gcd(in_channels, out_channels)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=None,
            groups=g,
            dilation=dilation,
            activation=activation,
        )


class _MultiScaleExtractor(nn.Module):
    """Two-stage depthwise multi-scale stack + 1x1 fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.stage1_k3 = _DepthwiseConv(channels, channels, kernel_size=3)
        self.stage1_k5 = _DepthwiseConv(channels, channels, kernel_size=5)
        self.stage2_k3 = _DepthwiseConv(2 * channels, channels, kernel_size=3)
        self.stage2_k5 = _DepthwiseConv(2 * channels, channels, kernel_size=5)
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.ch_norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise norm per spatial location (equivalent to LN on NHWC).
        x = self.ch_norm(x)
        s1_3 = self.stage1_k3(x)
        s1_5 = self.stage1_k5(x)
        s1 = torch.cat([s1_3, s1_5], dim=1)
        s2_3 = self.stage2_k3(s1)
        s2_5 = self.stage2_k5(s1)
        s2 = torch.cat([s2_3, s2_5], dim=1)
        return self.fuse(s2)


class _MaskLayer(nn.Module):
    """Channel-softmax mask over depthwise k1/k3/k5 branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pw = nn.Conv2d(
            channels, channels, kernel_size=1, groups=channels, bias=False,
        )
        self.dw3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False,
        )
        self.dw5 = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = F.softmax(self.pw(x) + self.dw3(x) + self.dw5(x), dim=1)
        return x * m


class SMFF(nn.Module):
    """Fuse aligned encoder and decoder features with SMMM-style masking.

    ``encoder_feature`` and ``decoder_feature`` must be ``(B, C, H, W)`` with the
    same ``B, C, H, W``. Returns a tensor of the same shape.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.enc_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dec_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.extractor = _MultiScaleExtractor(channels)
        self.mask = _MaskLayer(channels)
        self.dilated = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2, bias=False,
        )
        gn = _group_norm_groups(channels, 4)
        self.gn = nn.GroupNorm(gn, channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(
        self,
        encoder_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.enc_proj(encoder_feature)
        dec = self.dec_proj(decoder_feature)
        enc_ms = self.extractor(enc)
        dec_ms = self.extractor(dec)
        fused = self.mask(enc_ms) + self.mask(dec_ms)
        out = self.out_proj(self.gn(self.dilated(fused)))
        return out


# Paper abbreviation (AAAI 2026)
SMMM = SMFF

__all__ = ["SMFF", "SMMM"]


if __name__ == "__main__":
    c = 32
    x = torch.randn(2, c, 50, 50)
    y = torch.randn(2, c, 50, 50)
    m = SMFF(c)
    z = m(x, y)
    print("encoder:", x.shape, "decoder:", y.shape, "out:", z.shape)
