"""Strip-Guided Feature Refinement Module (SGFR).

A lightweight, plug-and-play block that refines feature maps by capturing
**local texture** (3x3 depthwise), **horizontal stripe** (1xk depthwise),
and **vertical stripe** (kx1 depthwise) patterns, fusing them through a
pointwise convolution, applying a learned spatial gate, and outputting a
residual-scaled refinement.

Designed for SAR oil-spill segmentation where elongated dark-patch
structures (oil slicks, ship wakes) benefit from directional context.

Pipeline
--------
1. 3x3 DWConv + BN + ReLU        -- local texture
2. 1xk DWConv + BN + ReLU        -- horizontal stripe enhancement
3. kx1 DWConv + BN + ReLU        -- vertical stripe enhancement
4. 1x1 Conv  + BN + ReLU         -- channel fusion
5. Spatial Gate (DW3x3 -> BN -> ReLU -> 1x1 -> Sigmoid)
6. Residual Scaling: out = x + gamma * (M * U)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SGFR(nn.Module):
    """Strip-Guided Feature Refinement Module.

    Args:
        channels: Channel width of the input feature map.
        stripe_kernel_size: Kernel length for the 1xk / kx1 stripe
            depthwise convolutions.  Must be an odd integer >= 3.
    """

    def __init__(self, channels: int, stripe_kernel_size: int = 7) -> None:
        super().__init__()
        assert stripe_kernel_size >= 3 and stripe_kernel_size % 2 == 1, (
            f"stripe_kernel_size must be an odd integer >= 3, got {stripe_kernel_size}"
        )
        k = stripe_kernel_size

        # Step 1 -- local texture: 3x3 depthwise
        self.local_dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False,
        )
        self.local_bn = nn.SyncBatchNorm(channels)

        # Step 2 -- horizontal stripe: 1xk depthwise
        self.h_dw = nn.Conv2d(
            channels, channels, kernel_size=(1, k), padding=(0, k // 2),
            groups=channels, bias=False,
        )
        self.h_bn = nn.SyncBatchNorm(channels)

        # Step 3 -- vertical stripe: kx1 depthwise
        self.v_dw = nn.Conv2d(
            channels, channels, kernel_size=(k, 1), padding=(k // 2, 0),
            groups=channels, bias=False,
        )
        self.v_bn = nn.SyncBatchNorm(channels)

        # Step 4 -- channel fusion: pointwise 1x1
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pw_bn = nn.SyncBatchNorm(channels)

        self.act = nn.ReLU(inplace=True)

        # Step 5 -- spatial gate
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.SyncBatchNorm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Step 6 -- residual scaling
        self.gamma = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sequential: local -> horizontal -> vertical -> pointwise
        u = self.act(self.local_bn(self.local_dw(x)))
        u = self.act(self.h_bn(self.h_dw(u)))
        u = self.act(self.v_bn(self.v_dw(u)))
        u = self.act(self.pw_bn(self.pw(u)))

        # spatial gate
        m = self.spatial_gate(u)                  # [B, 1, H, W]

        # residual output
        return x + self.gamma * (m * u)


__all__ = ["SGFR"]


if __name__ == "__main__":
    x = torch.randn(2, 256, 128, 128)
    module = SGFR(channels=256, stripe_kernel_size=7)
    y = module(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    assert x.shape == y.shape, "Shape mismatch!"

    total = sum(p.numel() for p in module.parameters())
    print(f"Params: {total / 1e3:.1f}K")

    y.sum().backward()
    print("Backward OK")
