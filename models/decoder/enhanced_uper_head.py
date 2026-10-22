"""Enhanced UPerNet decode head.

Combines three standalone modules into a drop-in replacement of the original
``UPerHead``:

- ``LGDA`` is applied to each shallow lateral (C1/C2/C3) after the 1x1
  channel-alignment convolution to denoise low-level SAR features.
- ``SPCM`` replaces the original ``PPM + bottleneck`` on the deepest stage
  (C4) and outputs ``channels`` channels directly.
- ``ClassAwareGatedFPN`` replaces the top-down FPN, ``fpn_convs`` and
  ``fpn_bottleneck`` with a single fused stream that already returns a
  feature map at the finest resolution.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.class_aware_fpn import ClassAwareGatedFPN
from models.modules.lgda import LGDA
from models.modules.oldm import OLDM
from models.modules.spcm import SPCM


class EnhancedUPerHead(nn.Module):
    """Enhanced UPerNet decode head with LGDA + SPCM + ClassAwareGatedFPN."""

    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384, 768),
        channels: int = 512,
        num_classes: int = 5,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        strip_kernel_size: int = 7,
        use_local_branch: bool = True,
        use_diagonal_strips: bool = False,
        lgda_num_groups: int = 16,
        lgda_spatial_kernel_size: int = 7,
        lgda_channel_kernel_size: int = 5,
        dropout_ratio: float = 0.1,
        proto_contrast: dict | None = None,
        use_oldm: bool = False,
        oldm_reduction: int = 16,
    ) -> None:
        super().__init__()

        if len(in_channels) < 2:
            raise ValueError("in_channels must have at least 2 levels")

        self.in_channels = tuple(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.proto_contrast_enabled = bool(
            proto_contrast and proto_contrast.get("enabled", False)
        )

        self.lateral_convs = nn.ModuleList()
        self.lgda_modules = nn.ModuleList()
        for ch in self.in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, channels, kernel_size=1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.lgda_modules.append(
                LGDA(
                    channels=channels,
                    num_groups=lgda_num_groups,
                    spatial_kernel_size=lgda_spatial_kernel_size,
                    channel_kernel_size=lgda_channel_kernel_size,
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

        # Build ClassAwareGatedFPN with optional proto-contrast kwargs
        ca_fpn_kwargs: dict = dict(
            channels=channels,
            num_classes=num_classes,
            num_levels=len(self.in_channels),
        )
        if self.proto_contrast_enabled:
            assert proto_contrast is not None
            ca_fpn_kwargs.update(
                use_proto_contrast=True,
                proto_momentum=proto_contrast.get("momentum", 0.999),
                proto_temperature=proto_contrast.get("temperature", 0.1),
                proto_inter_margin=proto_contrast.get("inter_margin", 0.3),
                proto_confuse_margin=proto_contrast.get("confuse_margin", 0.5),
                confuse_pairs=proto_contrast.get("confuse_pairs", [[1, 2]]),
            )
        self.ca_fpn = ClassAwareGatedFPN(**ca_fpn_kwargs)

        self.oldm = OLDM(channels, reduction=oldm_reduction) if use_oldm else None

        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        nn.init.normal_(self.cls_seg.weight, mean=0.0, std=0.01)
        if self.cls_seg.bias is not None:
            nn.init.zeros_(self.cls_seg.bias)

    def forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        output_size: torch.Size | None = None,
        masks: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """
        Args:
            inputs: Multi-scale backbone features (C1..C4).
            output_size: If given, upsample logits to this spatial size.
            masks: GT label map ``(B, H, W)`` — only used during training
                when ``proto_contrast`` is enabled.

        Returns:
            ``(logits, enhanced_c3)`` when proto-contrast is off or at
            inference time; ``(logits, enhanced_c3, proto_losses)`` during
            training when proto-contrast is active.
        """
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f"expected {len(self.in_channels)} feature maps, got {len(inputs)}"
            )

        laterals: list[torch.Tensor] = []
        for i in range(len(self.in_channels) - 1):
            feat = self.lateral_convs[i](inputs[i])
            feat = self.lgda_modules[i](feat)
            laterals.append(feat)

        enhanced_c3 = laterals[-1]

        laterals.append(self.spcm(inputs[-1]))

        fpn_out = self.ca_fpn(laterals, masks=masks)

        proto_losses: dict[str, torch.Tensor] | None = None
        if isinstance(fpn_out, tuple):
            fused, proto_losses = fpn_out
        else:
            fused = fpn_out

        fused = self.dropout(fused)
        if self.oldm is not None:
            fused = self.oldm(fused)
        logits = self.cls_seg(fused)

        if output_size is not None:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        if proto_losses is not None:
            return logits, enhanced_c3, proto_losses
        return logits, enhanced_c3
