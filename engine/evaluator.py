from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from engine.inference import slide_inference
from utils.dist import is_dist_avail_and_initialized


def intersect_and_union(pred: torch.Tensor, label: torch.Tensor, num_classes: int):
    mask = label != 255
    pred = pred[mask]
    label = label[mask]
    intersect = pred[pred == label]
    area_intersect = torch.bincount(intersect, minlength=num_classes).float()
    area_pred = torch.bincount(pred, minlength=num_classes).float()
    area_label = torch.bincount(label, minlength=num_classes).float()
    area_union = area_pred + area_label - area_intersect
    return area_intersect, area_union, area_label


@torch.no_grad()
def evaluate_miou(
    model,
    dataloader,
    device,
    num_classes: int,
    slide_cfg: dict | None = None,
):
    """Evaluate mIoU / mDice with optional sliding-window inference.

    Per-class Dice is ``2 * TP / (2*TP + FP + FN)`` (macro from summed
    confusion counts), reported in percent like IoU.

    Args:
        slide_cfg: if not *None*, must contain ``crop_size`` and ``stride``
            keys.  Optional ``aggregate``: ``mean_logits`` (default) or
            ``vote`` for majority-vote fusion.  When provided, each image is
            evaluated via sliding-window inference instead of whole-image forward.
    """
    model.eval()
    total_inter = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_label = torch.zeros(num_classes, device=device)
    total_correct = torch.zeros(1, device=device)
    total_pixel = torch.zeros(1, device=device)

    for batch in dataloader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        if slide_cfg is not None:
            logits = slide_inference(
                model, images,
                crop_size=tuple(slide_cfg['crop_size']),
                stride=tuple(slide_cfg['stride']),
                num_classes=num_classes,
                aggregate=str(slide_cfg.get('aggregate', 'mean_logits')),
            )
        else:
            logits = model(images)['logits']
        preds = torch.argmax(logits, dim=1)

        for i in range(preds.shape[0]):
            inter, union, label_hist = intersect_and_union(
                preds[i], masks[i], num_classes)
            total_inter += inter
            total_union += union
            total_label += label_hist

        valid = masks != 255
        total_correct += (preds[valid] == masks[valid]).sum()
        total_pixel += valid.sum()

    # Aggregate statistics across all DDP ranks so that metrics reflect
    # the full validation set, not just each rank's shard.
    if is_dist_avail_and_initialized():
        dist.all_reduce(total_inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_union, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_label, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_pixel, op=dist.ReduceOp.SUM)

    inter_np = total_inter.cpu().numpy()
    union_np = total_union.cpu().numpy()
    label_np = total_label.cpu().numpy()
    # ``union = pred + label - inter``  =>  ``pred = union - label + inter``
    pred_np = union_np - label_np + inter_np

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = inter_np / union_np * 100
        acc = inter_np / label_np * 100
        denom = pred_np + label_np
        dice = np.where(denom > 0, 2.0 * inter_np / denom, np.nan) * 100
    miou = float(np.nanmean(iou))
    macc = float(np.nanmean(acc))
    mdice = float(np.nanmean(dice))
    aacc = float(total_correct.item() / max(total_pixel.item(), 1) * 100)
    return {
        'mIoU': miou,
        'mAcc': macc,
        'mDice': mdice,
        'aAcc': aacc,
        'IoU': iou,
        'Acc': acc,
        'Dice': dice,
    }
