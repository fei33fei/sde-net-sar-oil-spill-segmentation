"""UPerHead with SGFR on shallow laterals but standard FPN (no DG-FPN/OLDM).

Data flow:
    C1,C2,C3 -> lateral 1x1 -> SGFR -> L1,L2,L3
    C4 -> PPM or SG-PPM -> P4
    Standard top-down FPN + fpn_convs (unchanged from UPerHead)
    Align to P1 -> concat -> fpn_bottleneck -> dropout -> cls_seg
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.uper_head import UPerHead
from models.modules.sgfr import SGFR


class UPerHeadSGFR(UPerHead):
    """SGFR + standard FPN decode head (optionally with SG-PPM).

    Args:
        in_channels: Per-stage channel widths from backbone.
        channels: Unified channel width after lateral projection.
        num_classes: Number of semantic classes.
        pool_scales: PPM pooling scales.
        dropout_ratio: Dropout ratio before classification.
        sgfr_stripe_k: Strip kernel size for SGFR modules.
        use_sgppm: Use Scale-Gated PPM instead of vanilla PPM.
        sgppm_reduction: Channel reduction for SG-PPM gate.
    """

    def __init__(
        self,
        in_channels=(96, 192, 384, 768),
        channels=512,
        num_classes=5,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        sgfr_stripe_k: int = 7,
        use_sgppm: bool = False,
        sgppm_reduction: int = 16,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            dropout_ratio=dropout_ratio,
            use_sgppm=use_sgppm,
            sgppm_reduction=sgppm_reduction,
        )
        num_shallow = len(in_channels) - 1
        self.sgfr_modules = nn.ModuleList(
            SGFR(channels, stripe_kernel_size=sgfr_stripe_k)
            for _ in range(num_shallow)
        )

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        output_size: torch.Size | None = None,
    ) -> torch.Tensor:
        # Lateral projections + SGFR for C1, C2, C3
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            lat = lateral_conv(inputs[i])
            lat = self.sgfr_modules[i](lat)
            laterals.append(lat)

        # P4 from PPM or SG-PPM
        laterals.append(self.psp_forward(inputs[-1]))

        # Standard top-down FPN (same as UPerHead)
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear',
                align_corners=False)

        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)
        ]
        fpn_outs.append(laterals[-1])

        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=False)

        feats = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(feats)
        output = self.dropout(output)
        output = self.cls_seg(output)
        if output_size is not None:
            output = F.interpolate(
                output, size=output_size, mode='bilinear',
                align_corners=False)
        return output
