"""DySample-Gated Feature Pyramid Network (DG-FPN).

Top-down pathway that replaces standard bilinear upsampling with DySample
(content-aware dynamic upsampling, ICCV 2023) and uses learned sigmoid
gates for feature fusion at each level -- same gating formula as
``GatedTopDownFusion`` but with DySample instead of ``F.interpolate``.

    deep_up  = DySample(P_{i})
    gate     = sigmoid(Conv1x1([L_{i-1}, deep_up]))
    P_{i-1}  = L_{i-1} + gate * deep_up
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.dy_sample import DySample


class DGFPN(nn.Module):
    """DySample-Gated FPN top-down pathway.

    Args:
        channels: Unified channel width of every input lateral / PPM output.
        num_levels: Number of pyramid levels (typically 4: L1..L3 + P4).
    """

    def __init__(self, channels: int, num_levels: int = 4) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be >= 2")
        self.num_levels = num_levels
        n_fuse = num_levels - 1

        self.upsamplers = nn.ModuleList(
            DySample(in_channels=channels, scale=2) for _ in range(n_fuse)
        )
        self.gate_convs = nn.ModuleList(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
            for _ in range(n_fuse)
        )

    def forward(self, laterals: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply DySample-gated top-down fusion **in-place** on *laterals*.

        Args:
            laterals: ``[L1, L2, L3, P4]`` (shallow → deep), each
                ``(B, C, H_i, W_i)`` with spatial size halving per level.

        Returns:
            The same list, now containing ``[P1, P2, P3, P4]``.
        """
        for i in range(self.num_levels - 1, 0, -1):
            deep_up = self.upsamplers[i - 1](laterals[i])
            target_h, target_w = laterals[i - 1].shape[2:]
            if deep_up.shape[2:] != (target_h, target_w):
                deep_up = F.interpolate(
                    deep_up, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )
            gate = torch.sigmoid(
                self.gate_convs[i - 1](torch.cat([laterals[i - 1], deep_up], dim=1))
            )
            laterals[i - 1] = laterals[i - 1] + gate * deep_up
        return laterals


__all__ = ["DGFPN"]


if __name__ == "__main__":
    B, C = 2, 512
    feats = [
        torch.randn(B, C, 128, 128),
        torch.randn(B, C, 64, 64),
        torch.randn(B, C, 32, 32),
        torch.randn(B, C, 16, 16),
    ]
    module = DGFPN(channels=C, num_levels=4)
    out = module(feats)
    for i, o in enumerate(out):
        print(f"P{i+1}: {o.shape}")

    total = sum(p.numel() for p in module.parameters())
    print(f"Params: {total / 1e3:.1f}K")

    sum(o.sum() for o in out).backward()
    print("Backward OK")
