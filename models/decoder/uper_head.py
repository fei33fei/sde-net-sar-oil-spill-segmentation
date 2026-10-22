from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):

    def __init__(self, in_channels: int, channels: int, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = nn.ModuleList()
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, 1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True)))

    def forward(self, x: torch.Tensor):
        outs = []
        for ppm in self.ppm:
            out = ppm(x)
            out = F.interpolate(
                out, size=x.shape[2:], mode='bilinear', align_corners=False)
            outs.append(out)
        return outs


class UPerHead(nn.Module):

    def __init__(self,
                 in_channels=(96, 192, 384, 768),
                 channels=512,
                 num_classes=5,
                 pool_scales=(1, 2, 3, 6),
                 dropout_ratio=0.1,
                 use_sgppm: bool = False,
                 sgppm_reduction: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.use_sgppm = use_sgppm

        if use_sgppm:
            from models.decoder.sgppm import SGPPM
            self.sgppm = SGPPM(
                in_channels[-1], channels, pool_scales,
                align_corners=False, reduction=sgppm_reduction,
            )
            self.psp_modules = None
            self.bottleneck = None
        else:
            self.sgppm = None
            self.psp_modules = PPM(in_channels[-1], channels, pool_scales)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(
                    in_channels[-1] + len(pool_scales) * channels,
                    channels,
                    3,
                    padding=1,
                    bias=False),
                nn.SyncBatchNorm(channels),
                nn.ReLU(inplace=True))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ch in in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, channels, 1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True)))
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.SyncBatchNorm(channels),
                    nn.ReLU(inplace=True)))

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                len(in_channels) * channels, channels, 3, padding=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        nn.init.normal_(self.cls_seg.weight, mean=0, std=0.01)
        if self.cls_seg.bias is not None:
            nn.init.zeros_(self.cls_seg.bias)

    def psp_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_sgppm:
            return self.sgppm(x)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def forward(self,
                inputs: tuple[torch.Tensor, ...],
                output_size: torch.Size | None = None) -> torch.Tensor:
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs[-1]))

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=False)

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
                output, size=output_size, mode='bilinear', align_corners=False)
        return output
