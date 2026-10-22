"""UPerHead with gated top-down fusion and optional OLDM.

Inherits the vanilla ``UPerHead`` and replaces the unconditional
element-wise addition in the top-down pathway with ``GatedTopDownFusion``.
All other FPN behaviour (PPM, lateral convs, fpn_convs, fpn_bottleneck)
remains identical to the baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.uper_head import UPerHead
from models.modules.gated_top_down_fusion import GatedTopDownFusion
from models.modules.oldm import OLDM


class UPerHeadOLDMGatedTopDown(UPerHead):
    """UPerHead + GatedTopDownFusion + OLDM.

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
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            dropout_ratio=dropout_ratio,
        )
        self.gated_td = GatedTopDownFusion(channels, num_levels=len(in_channels))
        self.oldm = OLDM(channels, reduction=oldm_reduction) if use_oldm else None

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        output_size: torch.Size | None = None,
    ) -> torch.Tensor:
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs[-1]))

        self.gated_td(laterals)

        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)
        ]
        fpn_outs.append(laterals[-1])

        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        feats = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(feats)
        output = self.dropout(output)

        if self.oldm is not None:
            output = self.oldm(output)

        output = self.cls_seg(output)
        if output_size is not None:
            output = F.interpolate(
                output, size=output_size, mode="bilinear", align_corners=False,
            )
        return output
