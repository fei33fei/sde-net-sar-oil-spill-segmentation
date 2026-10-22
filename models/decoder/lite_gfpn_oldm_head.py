"""UPerNet-style decode head with LiteGFPN fusion and optional OLDM.

Replaces the vanilla top-down addition + fpn_bottleneck in ``UPerHead``
with the ``LiteGFPN`` module (GFM + gated top-down + learnable scale
weights) while keeping PPM on the deepest backbone stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.uper_head import PPM
from models.modules.lite_gfpn import LiteGFPN
from models.modules.oldm import OLDM


class LiteGFPNOLDMHead(nn.Module):
    """PPM + LiteGFPN + OLDM decode head.

    Args:
        in_channels: Per-stage channel widths from backbone.
        channels: Unified channel width after lateral projection.
        num_classes: Number of semantic classes.
        pool_scales: PPM pooling scales.
        dropout_ratio: Dropout ratio before classification.
        use_oldm: Whether to enable the OLDM branch.
        oldm_reduction: Channel reduction ratio inside OLDM.
    """

    def __init__(
        self,
        in_channels=(96, 192, 384, 768),
        channels=512,
        num_classes=5,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        use_oldm: bool = True,
        oldm_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # PPM on deepest stage → channels
        self.psp_modules = PPM(in_channels[-1], channels, pool_scales)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels[-1] + len(pool_scales) * channels,
                channels, 3, padding=1, bias=False,
            ),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
        )

        # Lateral 1x1 projections for C1..C(n-1)
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, channels, 1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.lite_gfpn = LiteGFPN(
            channels=channels,
            num_levels=len(in_channels),
        )

        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )
        self.oldm = OLDM(channels, reduction=oldm_reduction) if use_oldm else None

        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        nn.init.normal_(self.cls_seg.weight, mean=0, std=0.01)
        if self.cls_seg.bias is not None:
            nn.init.zeros_(self.cls_seg.bias)

    def psp_forward(self, x: torch.Tensor) -> torch.Tensor:
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        output_size: torch.Size | None = None,
    ) -> torch.Tensor:
        laterals = [
            self.lateral_convs[i](inputs[i])
            for i in range(len(self.in_channels) - 1)
        ]
        laterals.append(self.psp_forward(inputs[-1]))

        fused = self.lite_gfpn(laterals)
        fused = self.dropout(fused)

        if self.oldm is not None:
            fused = self.oldm(fused)

        logits = self.cls_seg(fused)
        if output_size is not None:
            logits = F.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False,
            )
        return logits
