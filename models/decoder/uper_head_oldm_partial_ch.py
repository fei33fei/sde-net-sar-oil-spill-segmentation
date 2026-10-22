"""UPerHead + PartialAttentionCH + OLDM.

Inherits the vanilla ``UPerHead`` and inserts ``PartialAttentionCH``
between ``fpn_bottleneck`` and ``dropout``, followed by ``OLDM`` before
``cls_seg``.  All FPN behaviour (PPM, lateral convs, top-down, fpn_convs,
fpn_bottleneck) remains identical to the baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.uper_head import UPerHead
from models.modules.oldm import OLDM
from models.modules.partial_attention_ch import PartialAttentionCH


class UPerHeadOLDMPartialCH(UPerHead):
    """UPerHead + PartialAttentionCH + OLDM.

    Args:
        in_channels: Per-stage channel widths from backbone.
        channels: Unified channel width after lateral projection.
        num_classes: Number of semantic classes.
        pool_scales: PPM pooling scales.
        dropout_ratio: Dropout ratio before classification.
        use_oldm: Whether to enable the OLDM branch.
        oldm_reduction: Channel reduction ratio inside OLDM.
        use_partial_ch: Whether to enable the PartialAttentionCH branch.
        partial_ch_split_factor: Split factor for PartialAttentionCH.
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
        use_partial_ch: bool = True,
        partial_ch_split_factor: int = 4,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            pool_scales=pool_scales,
            dropout_ratio=dropout_ratio,
        )
        self.partial_ch = (
            PartialAttentionCH(channels, split_factor=partial_ch_split_factor)
            if use_partial_ch else None
        )
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

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False,
            )

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

        if self.partial_ch is not None:
            output = self.partial_ch(output)

        output = self.dropout(output)

        if self.oldm is not None:
            output = self.oldm(output)

        output = self.cls_seg(output)
        if output_size is not None:
            output = F.interpolate(
                output, size=output_size, mode="bilinear", align_corners=False,
            )
        return output
