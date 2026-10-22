import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0.1) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        nn.init.normal_(self.cls_seg.weight, mean=0, std=0.01)
        if self.cls_seg.bias is not None:
            nn.init.zeros_(self.cls_seg.bias)

    def forward(self,
                x: torch.Tensor,
                output_size: torch.Size | None = None) -> torch.Tensor:
        x = self.convs(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        if output_size is not None:
            x = F.interpolate(
                x, size=output_size, mode='bilinear', align_corners=False)
        return x
