"""Class-Aware Gated FPN (standalone, not wired into UPerHead by default).

Expects pre-aligned multi-scale features (same ``channels``), e.g. after
external lateral 1x1, optional LGDA on shallow levels, and SPCM on C4.

When ``use_proto_contrast`` is enabled (training only), the module also:

1. Maintains **running prototypes** via EMA from per-batch GT-masked features.
2. Computes an **inter-prototype margin loss** that pushes all class
   prototypes apart, with a higher margin for user-specified confusing
   class pairs (e.g. oil_spill / look_alike).
3. Computes a **pixel-prototype alignment loss** (InfoNCE-style) that
   pulls each pixel embedding towards its GT prototype and away from others.

These losses are returned alongside the fused feature map during training
and are zero-cost at inference time.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROTO_EMBED_DIM = 64
_PROTO_SAMPLE_PIXELS = 1024


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        ),
        nn.SyncBatchNorm(out_channels),
        nn.ReLU(inplace=True),
    )


class ClassAwareGatedFPN(nn.Module):
    """GFM + gated top-down + weighted multi-scale fusion + class prototypes.

    Args:
        channels: Channel width of every input lateral (must match).
        num_classes: Number of semantic classes (default 5 for SAR oil spill).
        num_levels: Number of pyramid levels (typically 4: C1..C4).
        use_proto_contrast: Enable running-prototype EMA and contrastive
            losses (default ``False`` — existing behaviour unchanged).
        proto_momentum: EMA decay for running prototypes.
        proto_temperature: Temperature for pixel-proto alignment loss.
        proto_inter_margin: Minimum cosine-distance margin between any two
            class prototypes.
        proto_confuse_margin: Higher margin applied to ``confuse_pairs``.
        confuse_pairs: List of ``(cls_a, cls_b)`` pairs requiring
            ``proto_confuse_margin`` instead of ``proto_inter_margin``.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int = 5,
        num_levels: int = 4,
        use_proto_contrast: bool = False,
        proto_momentum: float = 0.999,
        proto_temperature: float = 0.1,
        proto_inter_margin: float = 0.3,
        proto_confuse_margin: float = 0.5,
        confuse_pairs: list[list[int]] | None = None,
    ) -> None:
        super().__init__()

        if num_levels < 2:
            raise ValueError("num_levels must be >= 2")

        self.channels = channels
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.use_proto_contrast = use_proto_contrast

        # ---- standard FPN components (unchanged) ----
        self.gfm_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gfm_gamma = nn.Parameter(torch.tensor(1e-3))

        self.gate_convs = nn.ModuleList(
            [
                nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
                for _ in range(num_levels - 1)
            ]
        )

        self.fpn_convs = nn.ModuleList(
            [_conv_bn_relu(channels, channels, 3, 1) for _ in range(num_levels - 1)]
        )

        self.scale_weights = nn.Parameter(torch.zeros(num_levels))

        self.fusion = _conv_bn_relu(
            num_levels * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.class_protos = nn.Parameter(
            torch.randn(num_classes, channels) * 0.02
        )
        self.proto_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.delta = nn.Parameter(torch.tensor(1e-3))

        # ---- prototype contrast components (only when enabled) ----
        if use_proto_contrast:
            self.register_buffer(
                "running_protos",
                torch.zeros(num_classes, channels),
            )
            self.register_buffer(
                "proto_init_count",
                torch.zeros(num_classes),
            )
            self.proto_momentum = proto_momentum
            self.proto_temperature = proto_temperature
            self.proto_inter_margin = proto_inter_margin
            self.proto_confuse_margin = proto_confuse_margin
            self.confuse_pairs: list[tuple[int, int]] = [
                (int(p[0]), int(p[1])) for p in (confuse_pairs or [[1, 2]])
            ]
            self._confuse_set = set()
            for a, b in self.confuse_pairs:
                self._confuse_set.add((min(a, b), max(a, b)))

            self.proto_proj_head = nn.Sequential(
                nn.Linear(channels, channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels, _PROTO_EMBED_DIM, bias=False),
            )

    # ------------------------------------------------------------------ #
    # Standard FPN helpers (unchanged)
    # ------------------------------------------------------------------ #
    def _apply_gfm(self, lat: list[torch.Tensor]) -> None:
        top = lat[-1]
        gfm_vec = F.adaptive_avg_pool2d(top, output_size=1)
        gfm_attn = torch.sigmoid(self.gfm_proj(gfm_vec))
        g = self.gfm_gamma
        for i in range(self.num_levels - 1):
            lat[i] = lat[i] * (1.0 + g * gfm_attn)

    def _gated_top_down(self, lat: list[torch.Tensor]) -> None:
        for i in range(self.num_levels - 1, 0, -1):
            deep_up = F.interpolate(
                lat[i],
                size=lat[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            gate_in = torch.cat([lat[i - 1], deep_up], dim=1)
            gate = torch.sigmoid(self.gate_convs[i - 1](gate_in))
            lat[i - 1] = lat[i - 1] + gate * deep_up

    def _class_prototype_enhance(self, fused: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fused.shape
        protos = self._effective_protos()
        flat = fused.flatten(2).transpose(1, 2)
        sim = flat @ protos.t() / math.sqrt(float(c))
        attn = F.softmax(sim, dim=-1)
        context = attn @ protos
        context = context.transpose(1, 2).reshape(b, c, h, w)
        context = self.proto_proj(context)
        return fused + self.delta * context

    def _effective_protos(self) -> torch.Tensor:
        """Return the prototypes used for attention.

        When proto-contrast is active and running prototypes have been
        initialized, blend learnable and running prototypes (equal weight).
        Otherwise fall back to learnable-only.
        """
        if not self.use_proto_contrast:
            return self.class_protos
        if not hasattr(self, "running_protos"):
            return self.class_protos
        initialized = self.proto_init_count > 0
        if not initialized.any():
            return self.class_protos
        rp = self.running_protos
        rp_norm = rp / (rp.norm(dim=1, keepdim=True) + 1e-6)
        lp_norm = self.class_protos / (
            self.class_protos.norm(dim=1, keepdim=True) + 1e-6
        )
        blended = 0.5 * lp_norm + 0.5 * rp_norm
        mask = initialized.unsqueeze(1).float()
        return self.class_protos * (1 - mask) + blended * self.channels ** 0.5 * mask

    # ------------------------------------------------------------------ #
    # Prototype contrast helpers (training only)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_running_protos(
        self, fused: torch.Tensor, masks: torch.Tensor
    ) -> None:
        """EMA-update running prototypes from batch features and GT masks."""
        b, c, h, w = fused.shape
        masks_down = F.interpolate(
            masks.float().unsqueeze(1),
            size=(h, w),
            mode="nearest",
        ).squeeze(1).long()

        flat = fused.detach().permute(0, 2, 3, 1).reshape(-1, c)
        labels = masks_down.reshape(-1)

        for cls in range(self.num_classes):
            cls_mask = labels == cls
            if cls_mask.sum() < 1:
                continue
            cls_mean = flat[cls_mask].mean(dim=0)
            if self.proto_init_count[cls] == 0:
                self.running_protos[cls] = cls_mean
                self.proto_init_count[cls] = 1
            else:
                m = self.proto_momentum
                self.running_protos[cls] = (
                    m * self.running_protos[cls] + (1 - m) * cls_mean
                )

    def _inter_proto_loss(self) -> torch.Tensor:
        """Hinge loss pushing all prototype pairs beyond a margin."""
        protos = self._effective_protos()
        protos_n = F.normalize(protos, dim=1)
        cos_sim = protos_n @ protos_n.t()
        loss = torch.tensor(0.0, device=protos.device)
        count = 0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                pair = (i, j)
                margin = (
                    self.proto_confuse_margin
                    if pair in self._confuse_set
                    else self.proto_inter_margin
                )
                dist = 1.0 - cos_sim[i, j]
                loss = loss + torch.clamp(margin - dist, min=0.0)
                count += 1
        return loss / max(count, 1)

    def _pixel_proto_align_loss(
        self, fused: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE-style alignment: each pixel should be closest to its GT
        prototype in the projected embedding space."""
        b, c, h, w = fused.shape
        masks_down = F.interpolate(
            masks.float().unsqueeze(1),
            size=(h, w),
            mode="nearest",
        ).squeeze(1).long()

        flat = fused.permute(0, 2, 3, 1).reshape(-1, c)
        labels = masks_down.reshape(-1)

        valid = labels != 255
        flat = flat[valid]
        labels = labels[valid]

        if flat.shape[0] == 0:
            return torch.tensor(0.0, device=fused.device)

        if flat.shape[0] > _PROTO_SAMPLE_PIXELS:
            idx = torch.randperm(flat.shape[0], device=flat.device)[
                :_PROTO_SAMPLE_PIXELS
            ]
            flat = flat[idx]
            labels = labels[idx]

        protos = self._effective_protos().detach()
        proj_pixels = F.normalize(self.proto_proj_head(flat), dim=1)
        proj_protos = F.normalize(self.proto_proj_head(protos), dim=1)

        logits = proj_pixels @ proj_protos.t() / self.proto_temperature
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        laterals: Sequence[torch.Tensor],
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            laterals: Multi-scale feature list.
            masks: GT label map ``(B, H, W)``, only needed during training
                when ``use_proto_contrast`` is enabled.

        Returns:
            ``fused`` tensor when ``masks`` is None or proto-contrast is off;
            ``(fused, proto_losses_dict)`` otherwise.
        """
        if len(laterals) != self.num_levels:
            raise ValueError(
                f"expected {self.num_levels} laterals, got {len(laterals)}"
            )

        lat = list(laterals)
        for i, t in enumerate(lat):
            if t.shape[1] != self.channels:
                raise ValueError(
                    f"lateral[{i}] has {t.shape[1]} channels, expected {self.channels}"
                )

        self._apply_gfm(lat)
        self._gated_top_down(lat)

        fpn_outs: list[torch.Tensor] = []
        for i in range(self.num_levels - 1):
            fpn_outs.append(self.fpn_convs[i](lat[i]))
        fpn_outs.append(lat[-1])

        target_hw = lat[0].shape[2:]
        for i in range(1, self.num_levels):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        w = F.softmax(self.scale_weights, dim=0)
        weighted = [w[i] * fpn_outs[i] for i in range(self.num_levels)]
        feats = torch.cat(weighted, dim=1)
        fused = self.fusion(feats)
        fused = self._class_prototype_enhance(fused)

        if not self.use_proto_contrast or masks is None:
            return fused

        self._update_running_protos(fused, masks)
        proto_losses = {
            "loss_inter_proto": self._inter_proto_loss(),
            "loss_pixel_align": self._pixel_proto_align_loss(fused, masks),
        }
        return fused, proto_losses


if __name__ == "__main__":
    laterals = [
        torch.randn(2, 32, 64, 64),
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 32, 8, 8),
    ]
    m = ClassAwareGatedFPN(channels=32, num_classes=5, num_levels=4)
    y = m(laterals)
    print("out:", y.shape)
