"""Improved UPerHead: SGFR on shallow laterals + SG-PPM on C4 + DG-FPN + OLDM.

Data flow:
    C1,C2,C3 → lateral 1×1 → SGFR → L1,L2,L3
    C4 → SG-PPM (or vanilla PPM) → P4
    [L1,L2,L3,P4] → DG-FPN (DySample + Gated fusion) → [P1,P2,P3,P4]
    Align to P1 → concat → fpn_bottleneck → dropout → OLDM → cls_seg
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.dg_fpn import DGFPN
from models.decoder.uper_head import UPerHead
from models.modules.oldm import OLDM
from models.modules.sgfr import SGFR


class UPerHeadSGFRDGFPNOLDM(UPerHead):
    """SGFR + SG-PPM + DG-FPN + OLDM decode head.

    Args:
        in_channels: Per-stage channel widths from backbone.
        channels: Unified channel width after lateral projection.
        num_classes: Number of semantic classes.
        pool_scales: PPM pooling scales.
        dropout_ratio: Dropout ratio before classification.
        sgfr_stripe_k: Strip kernel size for SGFR modules.
        use_oldm: Whether to enable the OLDM branch.
        oldm_reduction: Channel reduction ratio inside OLDM.
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
        use_oldm: bool = True,
        oldm_reduction: int = 16,
        use_sgppm: bool = True,
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
        # Remove inherited fpn_convs — DG-FPN replaces the standard FPN path
        del self.fpn_convs
        self.fpn_convs = None

        num_shallow = len(in_channels) - 1
        self.sgfr_modules = nn.ModuleList(
            SGFR(channels, stripe_kernel_size=sgfr_stripe_k)
            for _ in range(num_shallow)
        )
        self.dg_fpn = DGFPN(channels, num_levels=len(in_channels))
        self.oldm = OLDM(channels, reduction=oldm_reduction) if use_oldm else None

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

        # P4 from SG-PPM (or vanilla PPM)
        laterals.append(self.psp_forward(inputs[-1]))

        # DG-FPN: DySample + gated top-down fusion (in-place)
        self.dg_fpn(laterals)

        # Align all levels to P1 spatial size and concatenate
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i] = F.interpolate(
                laterals[i],
                size=laterals[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        feats = torch.cat(laterals, dim=1)
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
