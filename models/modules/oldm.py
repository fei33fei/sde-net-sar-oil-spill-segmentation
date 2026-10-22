"""Oil--Look-alike Disambiguation Module (OLDM).

A lightweight, plug-and-play feature-level disambiguation module designed
specifically for SAR oil-spill segmentation where *Oil Spill* (class 1) and
*Look-alike* (class 2) share very similar low-backscatter dark-patch
appearances.

OLDM operates **after** the GFPN fused feature and **before** ``cls_seg``.
It requires no GT masks, produces no extra losses or logits, and degrades
gracefully to an identity at initialisation (``gamma = 1e-3``, conservative
region gate, zero-initialised refine BN).

Pipeline
--------
1. Candidate-region gate  -- soft mask of Oil/Look-alike-likely pixels.
2. Oil-like / Look-like soft split  -- internal 2-way softmax (not a head).
3. Masked average pooling  -- extract per-class region context.
4. Difference-context MLP  -- channel gate from ``[oil, look, |diff|, prod]``.
5. Spatial gate  -- depthwise-conv on channel-modulated features.
6. Refine branch  -- lightweight residual delta.
7. Output  -- ``x + gamma * pair_region * spatial_gate * delta``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        act: nn.Module | None = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, groups=groups, bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x


def _masked_avg_pool(
    x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted spatial average pooling.

    Args:
        x: ``(B, C, H, W)`` feature map.
        mask: ``(B, 1, H, W)`` soft weight map (non-negative).

    Returns:
        ``(B, C, 1, 1)`` pooled context vector.
    """
    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x * mask).sum(dim=(2, 3), keepdim=True) / denom


class OLDM(nn.Module):
    """Oil--Look-alike Disambiguation Module.

    Args:
        channels: Channel width of the input fused feature.
        reduction: Channel reduction ratio for internal hidden layers.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)

        # Step 1 -- candidate-region gate  [B, 1, H, W]
        self.pair_region_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.pair_region_gate[3].bias, -2.0)

        # Step 2 -- oil-like / look-like soft split  [B, 2, H, W]
        self.pair_split_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )

        # Step 4 -- context MLP  4C -> hidden -> C
        self.context_mlp = nn.Sequential(
            nn.Conv2d(4 * channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Step 5 -- spatial gate  DWConv3x3 -> BN -> ReLU -> 1x1 -> Sigmoid
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Step 6 -- refine branch  Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.refine[4].weight)

        # Step 7 -- residual scaling
        self.gamma = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: candidate region
        pair_region = self.pair_region_gate(x)                # [B, 1, H, W]

        # Step 2: oil/look soft split
        pair_logits = self.pair_split_gate(x)                 # [B, 2, H, W]
        pair_prob = F.softmax(pair_logits, dim=1)             # [B, 2, H, W]
        oil_weight = pair_region * pair_prob[:, 0:1]          # [B, 1, H, W]
        look_weight = pair_region * pair_prob[:, 1:2]         # [B, 1, H, W]

        # Step 3: masked average pooling
        oil_context = _masked_avg_pool(x, oil_weight)         # [B, C, 1, 1]
        look_context = _masked_avg_pool(x, look_weight)       # [B, C, 1, 1]
        diff_context = (oil_context - look_context).abs()     # [B, C, 1, 1]
        prod_context = oil_context * look_context             # [B, C, 1, 1]
        context = torch.cat(
            [oil_context, look_context, diff_context, prod_context], dim=1,
        )                                                     # [B, 4C, 1, 1]

        # Step 4: channel gate
        channel_gate = self.context_mlp(context)              # [B, C, 1, 1]
        disc_feat = x * channel_gate                          # [B, C, H, W]

        # Step 5: spatial gate
        spatial = self.spatial_gate(disc_feat)                 # [B, 1, H, W]

        # Step 6: refine
        delta = self.refine(x)                                # [B, C, H, W]

        # Step 7: residual output (only within candidate region)
        out = x + self.gamma * pair_region * spatial * delta
        return out


if __name__ == "__main__":
    x = torch.randn(2, 256, 128, 128)
    module = OLDM(channels=256)
    y = module(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    assert x.shape == y.shape, "Shape mismatch!"

    total = sum(p.numel() for p in module.parameters())
    print(f"Params: {total / 1e3:.1f}K")

    y.sum().backward()
    print("Backward OK")
