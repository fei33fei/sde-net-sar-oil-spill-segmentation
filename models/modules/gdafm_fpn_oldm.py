"""GDAFM-FPN-OLDM: self-contained FPN neck with Gated Depthwise Attention
Fusion and Oil--Look-alike Disambiguation.

Replaces the element-wise addition in the standard UPerNet top-down pathway
with ``GDAFM`` — a lightweight spatial-gated module that adaptively blends
shallow detail and deep semantic features at each FPN level.

Input:  multi-scale backbone features ``(C1, C2, C3, C4)`` with distinct
        channel widths (e.g. 96 / 192 / 384 / 768).
Output: a single fused feature map ``(B, channels, H_C1, W_C1)`` ready for
        an external ``cls_seg`` 1x1 conv.

Pipeline
--------
1. Lateral 1x1 projection  -- C1/C2/C3 → ``channels``.
2. PPM + bottleneck on C4   -- → ``channels`` (reuses ``PPM`` from uper_head).
3. GDAFM top-down fusion    -- 3 independent GDAFM instances replace ``+``.
4. FPN 3x3 convs + concat   -- all levels upsampled to C1 resolution.
5. FPN bottleneck            -- ``4*channels`` → ``channels``.
6. Dropout + OLDM            -- optional disambiguation refinement.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.uper_head import PPM
from models.modules.oldm import OLDM


# ------------------------------------------------------------------ #
#  ConvBNAct helper (BatchNorm2d, matching GDAFM / OLDM convention)
# ------------------------------------------------------------------ #

class ConvBNAct(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, padding=padding,
                groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ------------------------------------------------------------------ #
#  GDAFM — Gated Depthwise Attention Fusion Module
# ------------------------------------------------------------------ #

class GDAFM(nn.Module):
    """Gated Depthwise Attention Fusion Module.

    Fuses a shallow/detail feature (``low``) with a deep/semantic feature
    (``high``) via a learned spatial gate derived from channel-wise mean/max
    statistics, depthwise-enhanced branches, and a shallow residual path.

    Intended as a drop-in replacement for ``P_i = L_i + Up(P_{i+1})``.
    """

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.low_proj = ConvBNAct(low_channels, out_channels, 1, 0)
        self.high_proj = ConvBNAct(high_channels, out_channels, 1, 0)

        self.attn = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.low_dw = nn.Sequential(
            ConvBNAct(out_channels, out_channels, 1, 0),
            ConvBNAct(out_channels, out_channels, 3, 1, groups=out_channels),
        )

        self.high_dw = nn.Sequential(
            ConvBNAct(out_channels, out_channels, 1, 0),
            ConvBNAct(out_channels, out_channels, 3, 1, groups=out_channels),
        )

        self.low_residual = ConvBNAct(out_channels, out_channels, 1, 0, act=False)
        self.out_proj = ConvBNAct(out_channels, out_channels, 1, 0)

    @staticmethod
    def _mean_max_map(x: torch.Tensor):
        return x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        if high.shape[2:] != low.shape[2:]:
            high = F.interpolate(
                high, size=low.shape[2:],
                mode="bilinear", align_corners=False,
            )

        low_hat = self.low_proj(low)
        high_hat = self.high_proj(high)

        low_mean, low_max = self._mean_max_map(low_hat)
        high_mean, high_max = self._mean_max_map(high_hat)
        alpha = self.attn(
            torch.cat([low_mean, low_max, high_mean, high_max], dim=1)
        )

        fused = (
            self.low_dw(low_hat) * alpha
            + self.high_dw(high_hat) * (1.0 - alpha)
            + self.low_residual(low_hat)
        )
        return self.out_proj(fused)


# ------------------------------------------------------------------ #
#  SyncBN helper (for FPN-level components, matching uper_head.py)
# ------------------------------------------------------------------ #

def _sync_conv_bn_relu(
    in_channels: int, out_channels: int, kernel_size: int, padding: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.SyncBatchNorm(out_channels),
        nn.ReLU(inplace=True),
    )


# ------------------------------------------------------------------ #
#  GDAFMFPNOLDM — self-contained FPN neck
# ------------------------------------------------------------------ #

class GDAFMFPNOLDM(nn.Module):
    """Self-contained FPN neck: lateral + PPM + GDAFM top-down + OLDM.

    Args:
        in_channels: Per-stage channel widths from backbone (e.g. [96,192,384,768]).
        channels: Unified channel width after lateral projection.
        pool_scales: PPM pooling scales.
        dropout_ratio: Dropout ratio before OLDM.
        use_oldm: Whether to enable the OLDM branch.
        oldm_reduction: Channel reduction ratio inside OLDM.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384, 768),
        channels: int = 512,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        dropout_ratio: float = 0.1,
        use_oldm: bool = True,
        oldm_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.channels = channels
        num_levels = len(in_channels)

        # -- lateral 1x1 projections for C1 .. C_{N-1} --
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels[:-1]:
            self.lateral_convs.append(
                _sync_conv_bn_relu(ch, channels, kernel_size=1, padding=0))

        # -- PPM on the deepest stage --
        self.psp_modules = PPM(in_channels[-1], channels, pool_scales)
        self.psp_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels[-1] + len(pool_scales) * channels,
                channels, 3, padding=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
        )

        # -- 3 independent GDAFM instances for top-down fusion --
        self.gdafm_modules = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.gdafm_modules.append(
                GDAFM(low_channels=channels, high_channels=channels,
                       out_channels=channels))

        # -- FPN 3x3 convs on the fused shallow levels --
        self.fpn_convs = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.fpn_convs.append(
                _sync_conv_bn_relu(channels, channels, kernel_size=3, padding=1))

        # -- bottleneck: concat all levels → channels --
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(num_levels * channels, channels, 3, padding=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity())

        self.oldm = OLDM(channels, reduction=oldm_reduction) if use_oldm else nn.Identity()

    def _psp_forward(self, x: torch.Tensor) -> torch.Tensor:
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        return self.psp_bottleneck(torch.cat(psp_outs, dim=1))

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            features: Multi-scale backbone outputs ``(C1, C2, ..., C_N)``.

        Returns:
            Fused feature ``(B, channels, H_C1, W_C1)`` ready for cls_seg.
        """
        num_levels = len(self.in_channels)

        # lateral projections for shallow stages + PPM for deepest
        laterals = [
            self.lateral_convs[i](features[i])
            for i in range(num_levels - 1)
        ]
        laterals.append(self._psp_forward(features[-1]))

        # --- GDAFM top-down: replace  lat[i-1] = lat[i-1] + Up(lat[i]) ---
        for i in range(num_levels - 1, 0, -1):
            laterals[i - 1] = self.gdafm_modules[i - 1](
                low=laterals[i - 1], high=laterals[i])

        # FPN 3x3 on shallow levels; deepest passes through directly
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(num_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        # upsample everything to the finest (C1) resolution and concat
        target_hw = fpn_outs[0].shape[2:]
        for i in range(1, num_levels):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_hw,
                mode="bilinear", align_corners=False)

        output = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        output = self.dropout(output)
        output = self.oldm(output)
        return output


# ------------------------------------------------------------------ #
#  Self-test
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    B = 2
    features = (
        torch.randn(B, 96, 128, 128),
        torch.randn(B, 192, 64, 64),
        torch.randn(B, 384, 32, 32),
        torch.randn(B, 768, 16, 16),
    )

    module = GDAFMFPNOLDM(
        in_channels=(96, 192, 384, 768),
        channels=256,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        use_oldm=True,
        oldm_reduction=16,
    )

    y = module(features)
    print("Input shapes :", [f.shape for f in features])
    print("Output shape :", y.shape)
    assert y.shape == (B, 256, 128, 128), f"Shape mismatch: {y.shape}"

    total = sum(p.numel() for p in module.parameters())
    print(f"Params: {total / 1e6:.2f}M")

    y.sum().backward()
    print("Backward OK")
