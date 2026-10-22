"""Partial Spatial-Attention block (PSAB) — partial-channel split with spatial gating.

From *Partial Channel Network* (AAAI 2026). Splits channels into an
``untouched`` branch (modulated by a 1×1 + Hardsigmoid map) and a
``conv`` branch (1×1 conv + norm). The attention map is applied to the
untouched branch; for element-wise consistency the spatial weight must
match the untouched channel count, so ``dim_untouched == dim_conv`` is
required (e.g. ``partial=0.5`` with an even ``dim``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PartialAttentionSA(nn.Module):
    """PSAB: partial split + hard-sigmoid spatial gating on the first group.

    Args:
        dim: Total channel count.
        partial: Fraction of channels assigned to the 1×1 ``conv`` branch
            (the rest form the attention-modulated branch). The two parts
            must have equal size for the gating to apply; this holds when
            ``partial=0.5`` (and ``dim`` is even), or more generally when
            ``int(partial * dim) == dim - int(partial * dim)``.
    """

    def __init__(self, dim: int, partial: float = 0.5) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be >= 2")
        if not 0.0 < partial < 1.0:
            raise ValueError("partial must be in (0, 1)")

        self.dim = dim
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv

        if self.dim_untouched != self.dim_conv:
            raise ValueError(
                "PartialAttentionSA requires the two channel groups to be equal "
                "in size (e.g. partial=0.5 and even dim) so the attention map "
                f"matches the untouched branch. Got dim_conv={self.dim_conv}, "
                f"dim_untouched={self.dim_untouched}."
            )

        self.conv = nn.Conv2d(
            self.dim_conv, self.dim_conv, kernel_size=1, bias=False,
        )
        self.conv_attn = nn.Conv2d(
            self.dim_untouched, self.dim_conv, kernel_size=1, bias=False,
        )
        self.norm = nn.SyncBatchNorm(self.dim_untouched)
        self.norm2 = nn.SyncBatchNorm(self.dim_conv)
        self.act = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], dim=1)
        weight = self.act(self.conv_attn(x1))
        x1 = self.norm(x1 * weight)
        x2 = self.norm2(self.conv(x2))
        return torch.cat((x1, x2), dim=1)


__all__ = ["PartialAttentionSA"]


if __name__ == "__main__":
    x = torch.rand(2, 32, 50, 50)
    m = PartialAttentionSA(dim=32, partial=0.5)
    y = m(x)
    print("in:", x.shape, "out:", y.shape)
