"""Sliding window inference for segmentation models.

Mirrors the logic of mmseg ``EncoderDecoder.slide_inference``: slide a
crop-sized window across the full image, then fuse overlapping predictions.

Two fusion modes:

* ``mean_logits`` — accumulate logits and divide by overlap count (default).
* ``vote`` — each window contributes one hard class vote per pixel; overlap
  uses majority vote (ties broken by smaller class index).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def slide_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    crop_size: tuple[int, int],
    stride: tuple[int, int],
    num_classes: int,
    aggregate: str = 'mean_logits',
) -> torch.Tensor:
    """Run sliding-window inference.

    Args:
        model: segmentation model whose forward returns ``{'logits': ...}``.
        images: ``(B, C, H, W)`` input tensor.
        crop_size: ``(h_crop, w_crop)`` window size.
        stride: ``(h_stride, w_stride)`` step between windows.
        num_classes: number of output classes.
        aggregate: ``mean_logits`` or ``vote`` (see module docstring).

    Returns:
        ``(B, num_classes, H, W)``. For ``mean_logits``, true averaged logits.
        For ``vote``, per-class vote counts; ``argmax(dim=1)`` is the fused label.
    """
    if aggregate not in ('mean_logits', 'vote'):
        raise ValueError(f"aggregate must be 'mean_logits' or 'vote', got {aggregate!r}")

    B, C, H, W = images.shape
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride

    h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1

    if aggregate == 'mean_logits':
        preds = images.new_zeros((B, num_classes, H, W))
        count_mat = images.new_zeros((B, 1, H, W))
    else:
        votes = images.new_zeros((B, num_classes, H, W))

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, H)
            x2 = min(x1 + w_crop, W)
            # Snap back so the window is always exactly crop_size
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            crop_img = images[:, :, y1:y2, x1:x2]
            crop_logits = model(crop_img)['logits']
            if aggregate == 'mean_logits':
                preds[:, :, y1:y2, x1:x2] += crop_logits
                count_mat[:, :, y1:y2, x1:x2] += 1
            else:
                pred_crop = crop_logits.argmax(dim=1).long()
                oh = F.one_hot(pred_crop, num_classes).permute(0, 3, 1, 2).to(
                    votes.dtype)
                votes[:, :, y1:y2, x1:x2] += oh

    if aggregate == 'mean_logits':
        return preds / count_mat
    return votes
