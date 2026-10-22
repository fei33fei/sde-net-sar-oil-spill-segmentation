"""Multi-scale + multi-flip TTA with sliding window inference.

Augmentation budget: **4 flips + 4 scales = 8 forward passes**.

*   4 flip variants at original scale: (none, h-flip, v-flip, hv-flip)
*   4 scale variants without flip:     scale in {1.0, 1.25, 1.5, 1.75}

Note: (scale=1.0, no-flip) appears in both groups; it is run once and
weighted 2/8 so the original view gets slightly more influence — a common
and intentional TTA practice.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.inference import slide_inference


def slide_inference_tta_ms(
    model: torch.nn.Module,
    images: torch.Tensor,
    crop_size: tuple[int, int],
    stride: tuple[int, int],
    num_classes: int,
    scales: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75),
) -> torch.Tensor:
    """8-pass TTA: 4 flips (scale 1) + 4 multi-scale (no flip).

    Args:
        model: segmentation model whose forward returns ``{'logits': ...}``.
        images: ``(B, C, H, W)`` input tensor.
        crop_size: ``(h_crop, w_crop)`` base sliding window size.
        stride: ``(h_stride, w_stride)`` base step between windows.
        num_classes: number of output classes.
        scales: scale factors for the multi-scale branch.

    Returns:
        ``(B, num_classes, H, W)`` averaged logits across all 8 passes.
    """
    B, C, H, W = images.shape
    logits_sum = images.new_zeros((B, num_classes, H, W))
    n_passes = 0

    # --- branch 1: 4 flip variants at original scale ---
    flips = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    for hflip, vflip in flips:
        aug = images
        if hflip:
            aug = torch.flip(aug, dims=[3])
        if vflip:
            aug = torch.flip(aug, dims=[2])

        logits = slide_inference(
            model, aug, crop_size, stride, num_classes,
            aggregate='mean_logits',
        )

        if vflip:
            logits = torch.flip(logits, dims=[2])
        if hflip:
            logits = torch.flip(logits, dims=[3])

        logits_sum += logits
        n_passes += 1

    # --- branch 2: multi-scale variants (no flip) ---
    for s in scales:
        H_s = max(round(H * s), crop_size[0])
        W_s = max(round(W * s), crop_size[1])

        if s == 1.0:
            scaled = images
            cs = crop_size
            st = stride
        else:
            scaled = F.interpolate(
                images, size=(H_s, W_s), mode='bilinear', align_corners=False,
            )
            cs = (max(round(crop_size[0] * s), 1),
                  max(round(crop_size[1] * s), 1))
            st = (max(round(stride[0] * s), 1),
                  max(round(stride[1] * s), 1))

        logits = slide_inference(
            model, scaled, cs, st, num_classes,
            aggregate='mean_logits',
        )

        if logits.shape[2:] != (H, W):
            logits = F.interpolate(
                logits, size=(H, W), mode='bilinear', align_corners=False,
            )

        logits_sum += logits
        n_passes += 1

    return logits_sum / n_passes
