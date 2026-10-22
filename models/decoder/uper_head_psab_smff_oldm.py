"""UPerHead with PSAB (C1-C3) + EnhancedPPM (C4) + SMFF FPN + OLDM.

Architecture
------------
1. Lateral 1x1 conv on C1/C2/C3 → ``channels``, each followed by
   ``PartialAttentionSA`` for noise-aware spatial gating.
2. ``EnhancedPPM`` on C4 replaces the standard PPM + bottleneck,
   providing 9-branch directional context extraction.
3. Top-down FPN uses ``SMFF`` at each level instead of element-wise add,
   performing structure-aware multi-scale fusion.
4. FPN 3x3 conv + concat + bottleneck + dropout.
5. ``OLDM`` disambiguates oil-spill vs look-alike before classification.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.enhanced_ppm import EnhancedPPM
from models.modules.oldm import OLDM
from models.modules.partial_attention_sa import PartialAttentionSA
from models.modules.SMFF import SMFF


class UPerHeadPSABSMFFOLDM(nn.Module):
    """UPerNet decode head: PSAB + EnhancedPPM + SMFF + OLDM.

    Args:
        in_channels: Per-stage channel widths from backbone.
        channels: Unified channel width after lateral projection.
        num_classes: Number of semantic classes.
        pool_scales: Scales for the EnhancedPPM classic PPM branches.
        diag_kernel_size: Kernel size for EnhancedPPM diagonal strip convs.
        partial_ratio: Channel split ratio for PartialAttentionSA.
        dropout_ratio: Dropout ratio before classification.
        use_oldm: Whether to enable the OLDM branch.
        oldm_reduction: Channel reduction ratio inside OLDM.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384, 768),
        channels: int = 512,
        num_classes: int = 5,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        diag_kernel_size: int = 7,
        partial_ratio: float = 0.5,
        dropout_ratio: float = 0.1,
        use_oldm: bool = True,
        oldm_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.channels = channels
        num_levels = len(in_channels)

        # --- Lateral 1x1 projections + PSAB for C1..C_{N-1} ---
        self.lateral_convs = nn.ModuleList()
        self.psab_modules = nn.ModuleList()
        for ch in in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, channels, 1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.psab_modules.append(
                PartialAttentionSA(dim=channels, partial=partial_ratio)
            )

        # --- EnhancedPPM on deepest stage (replaces PPM + bottleneck) ---
        self.enhanced_ppm = EnhancedPPM(
            in_channels=in_channels[-1],
            out_channels=channels,
            pool_scales=tuple(pool_scales),
            diag_kernel_size=diag_kernel_size,
        )

        # --- SMFF modules for top-down fusion (one per connection) ---
        self.smff_modules = nn.ModuleList(
            SMFF(channels) for _ in range(num_levels - 1)
        )

        # --- FPN 3x3 convs on fused levels ---
        self.fpn_convs = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # --- FPN bottleneck: concat all levels → channels ---
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(num_levels * channels, channels, 3, padding=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )

        # --- OLDM ---
        self.oldm = OLDM(channels, reduction=oldm_reduction) if use_oldm else None

        # --- Classification head ---
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        nn.init.normal_(self.cls_seg.weight, mean=0.0, std=0.01)
        if self.cls_seg.bias is not None:
            nn.init.zeros_(self.cls_seg.bias)

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        output_size: torch.Size | None = None,
    ) -> torch.Tensor:
        num_levels = len(self.in_channels)

        # 1) Lateral projections + PSAB for shallow stages
        laterals: list[torch.Tensor] = []
        for i in range(num_levels - 1):
            feat = self.lateral_convs[i](inputs[i])
            feat = self.psab_modules[i](feat)
            laterals.append(feat)

        # 2) EnhancedPPM on the deepest stage
        laterals.append(self.enhanced_ppm(inputs[-1]))

        # 3) Top-down SMFF fusion
        for i in range(num_levels - 1, 0, -1):
            deep_up = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = self.smff_modules[i - 1](
                encoder_feature=laterals[i - 1],
                decoder_feature=deep_up,
            )

        # 4) FPN 3x3 on shallow levels; deepest passes through
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(num_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        # Upsample all to finest resolution and concat
        target_hw = fpn_outs[0].shape[2:]
        for i in range(1, num_levels):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_hw, mode="bilinear", align_corners=False
            )

        output = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        output = self.dropout(output)

        # 5) OLDM disambiguation
        if self.oldm is not None:
            output = self.oldm(output)

        # 6) Classification
        output = self.cls_seg(output)
        if output_size is not None:
            output = F.interpolate(
                output, size=output_size, mode="bilinear", align_corners=False
            )
        return output
