"""Test-Time Augmentation (TTA) with sliding window inference.

Runs ``slide_inference`` on 4 flipped variants of the input image (original,
horizontal flip, vertical flip, HV flip), then averages the logits in the
original coordinate frame.  The result is a drop-in replacement for
``slide_inference`` that typically yields +0.3--1.0 mIoU at 4x inference cost.
"""

from __future__ import annotations

import torch

from engine.inference import slide_inference


def slide_inference_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    crop_size: tuple[int, int],
    stride: tuple[int, int],
    num_classes: int,
) -> torch.Tensor:
    """4-flip TTA over slide_inference, returns averaged logits.

    Args:
        model: segmentation model whose forward returns ``{'logits': ...}``.
        images: ``(B, C, H, W)`` input tensor.
        crop_size: ``(h_crop, w_crop)`` sliding window size.
        stride: ``(h_stride, w_stride)`` step between windows.
        num_classes: number of output classes.

    Returns:
        ``(B, num_classes, H, W)`` averaged logits across 4 flip variants.
    """
    flips = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    logits_sum: torch.Tensor | None = None
    for hflip, vflip in flips:
        aug = images
        if hflip:
            aug = torch.flip(aug, dims=[3])
        if vflip:
            aug = torch.flip(aug, dims=[2])

        logits = slide_inference(
            model, aug, crop_size, stride, num_classes, aggregate='mean_logits',
        )

        if vflip:
            logits = torch.flip(logits, dims=[2])
        if hflip:
            logits = torch.flip(logits, dims=[3])

        logits_sum = logits if logits_sum is None else logits_sum + logits

    return logits_sum / len(flips)
