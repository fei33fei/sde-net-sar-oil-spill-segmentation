"""DySample -- lightweight dynamic upsampler (ICCV 2023).

Ported from the official repository:
  https://github.com/tiny-smart/dysample  (MIT License)
  Paper: "Learning to Upsample by Learning to Sample"
         https://arxiv.org/abs/2308.15085

Reformulates upsampling as a point-sampling task using PyTorch's built-in
``grid_sample``.  No custom CUDA kernels required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Content-aware dynamic upsampler.

    Args:
        in_channels: Number of input feature channels.
        scale: Spatial upsampling factor (default 2).
        style: ``'lp'`` (linear then pixel-shuffle) or ``'pl'`` (pixel-shuffle
            then linear).
        groups: Number of sampling groups.
        dyscope: If True, use a learned scope factor to scale offsets.
    """

    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ("lp", "pl")
        if style == "pl":
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            offset_in = in_channels // scale ** 2
        else:
            offset_in = in_channels
        out_channels = 2 * groups * scale ** 2 if style == "lp" else 2 * groups

        self.offset = nn.Conv2d(offset_in, out_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)

        if dyscope:
            self.scope = nn.Conv2d(offset_in, out_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid(h, h, indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(W, dtype=x.dtype, device=x.device) + 0.5
        coords = (
            torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.style == "pl":
            x_ = F.pixel_shuffle(x, self.scale)
            if hasattr(self, "scope"):
                offset = (
                    F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale)
                    * 0.5
                    + self.init_pos
                )
            else:
                offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
            return self._sample(x, offset)
        # style == 'lp'
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)


__all__ = ["DySample"]


if __name__ == "__main__":
    x = torch.randn(2, 512, 16, 16)
    module = DySample(in_channels=512, scale=2)
    y = module(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    assert y.shape == (2, 512, 32, 32), f"Expected (2,512,32,32), got {y.shape}"

    total = sum(p.numel() for p in module.parameters())
    print(f"Params: {total / 1e3:.1f}K")

    y.sum().backward()
    print("Backward OK")
