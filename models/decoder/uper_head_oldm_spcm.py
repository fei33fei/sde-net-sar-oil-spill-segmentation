"""UPerHead with SPCM on the deepest stage (no LGDA) and OLDM.

Compared to ``UPerHeadOLDM`` (vanilla PPM + unconditional top-down + OLDM):

- **SPCM** replaces the original PPM + bottleneck on the deepest stage (C4).

Shallow laterals (C1–C3) use only the standard 1x1 lateral projection — **no LGDA**.

The FPN (top-down addition, fpn_convs, fpn_bottleneck) and post-fusion pipeline
(dropout -> OLDM -> cls_seg) match ``UPerHeadOLDMLgdaSpcm`` aside from LGDA.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.oldm import OLDM
from models.modules.spcm import SPCM


class UPerHeadOLDMSpcm(nn.Module):
    """SPCM + vanilla FPN + OLDM decode head (no LGDA).

    Args:
        in_channels: Per-stage channel widths from backbone.
        channels: Unified channel width after lateral projection.
        num_classes: Number of semantic classes.
        pool_scales: Pooling scales for SPCM pyramid branches.
        dropout_ratio: Dropout ratio before classification.
        use_oldm: Whether to enable the OLDM branch.
        oldm_reduction: Channel reduction ratio inside OLDM.
        strip_kernel_size: Strip kernel size for SPCM.
        use_local_branch: Whether to use SPCM local branch.
        use_diagonal_strips: Whether to use SPCM diagonal strips.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384, 768),
        channels: int = 512,
        num_classes: int = 5,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        dropout_ratio: float = 0.1,
        use_oldm: bool = True,
        oldm_reduction: int = 16,
        strip_kernel_size: int = 7,
        use_local_branch: bool = True,
        use_diagonal_strips: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.channels = channels

        self.lateral_convs = nn.ModuleList()
        for ch in self.in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, channels, 1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.spcm = SPCM(
            in_channels=self.in_channels[-1],
            out_channels=channels,
            pool_scales=tuple(pool_scales),
            strip_kernel_size=strip_kernel_size,
            use_local_branch=use_local_branch,
            use_diagonal_strips=use_diagonal_strips,
        )

        self.fpn_convs = nn.ModuleList()
        for _ in self.in_channels[:-1]:
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                len(in_channels) * channels, channels, 3, padding=1, bias=False,
            ),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )
        self.oldm = OLDM(channels, reduction=oldm_reduction) if use_oldm else None
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        nn.init.normal_(self.cls_seg.weight, mean=0.0, std=0.01)
        if self.cls_seg.bias is not None:
            nn.init.zeros_(self.cls_seg.bias)

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        output_size: torch.Size | None = None,
    ) -> torch.Tensor:
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.spcm(inputs[-1]))

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
        output = self.dropout(output)

        if self.oldm is not None:
            output = self.oldm(output)

        output = self.cls_seg(output)
        if output_size is not None:
            output = F.interpolate(
                output, size=output_size, mode="bilinear", align_corners=False,
            )
        return output
