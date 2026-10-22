"""Test script with 4-flip TTA + sliding window inference.

Usage mirrors ``tools/test.py`` but replaces standard slide inference with
TTA-enhanced slide inference.  Supports ``--save-md PATH`` to auto-append
results to a markdown file.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from datasets.oil_spill import OilSpillDataset
from datasets.transforms import EvalTransform
from engine.evaluator import intersect_and_union
from engine.inference_tta import slide_inference_tta
from tools.train import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def _build_slide_cfg(cfg, args):
    test_cfg = cfg.get('test', {})
    mode = test_cfg.get('mode', 'whole')
    if args.slide:
        mode = 'slide'
    if mode != 'slide':
        return None
    crop_size = list(test_cfg.get('crop_size', cfg['data']['crop_size']))
    overlap = args.overlap if args.overlap is not None else float(test_cfg.get('overlap', 0.2))
    stride = [int(c * (1 - overlap)) for c in crop_size]
    print(f'[slide-tta] crop_size={crop_size}, stride={stride}, overlap={overlap}')
    return {'crop_size': crop_size, 'stride': stride}


@torch.no_grad()
def evaluate_tta(model, dataloader, device, num_classes, slide_cfg):
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
            logits = slide_inference_tta(
                model, images,
                crop_size=tuple(slide_cfg['crop_size']),
                stride=tuple(slide_cfg['stride']),
                num_classes=num_classes,
            )
        else:
            flips = [(False, False), (True, False), (False, True), (True, True)]
            logits_sum = None
            for hflip, vflip in flips:
                aug = images
                if hflip:
                    aug = torch.flip(aug, dims=[3])
                if vflip:
                    aug = torch.flip(aug, dims=[2])
                out = model(aug)['logits']
                if vflip:
                    out = torch.flip(out, dims=[2])
                if hflip:
                    out = torch.flip(out, dims=[3])
                logits_sum = out if logits_sum is None else logits_sum + out
            logits = logits_sum / len(flips)

        preds = torch.argmax(logits, dim=1)
        for i in range(preds.shape[0]):
            inter, union, label_hist = intersect_and_union(preds[i], masks[i], num_classes)
            total_inter += inter
            total_union += union
            total_label += label_hist
        valid = masks != 255
        total_correct += (preds[valid] == masks[valid]).sum()
        total_pixel += valid.sum()

    inter_np = total_inter.cpu().numpy()
    union_np = total_union.cpu().numpy()
    label_np = total_label.cpu().numpy()
    pred_np = union_np - label_np + inter_np

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = inter_np / union_np * 100
        acc = inter_np / label_np * 100
        denom = pred_np + label_np
        dice = np.where(denom > 0, 2.0 * inter_np / denom, np.nan) * 100
    return {
        'mIoU': float(np.nanmean(iou)),
        'mAcc': float(np.nanmean(acc)),
        'mDice': float(np.nanmean(dice)),
        'aAcc': float(total_correct.item() / max(total_pixel.item(), 1) * 100),
        'IoU': iou,
        'Acc': acc,
        'Dice': dice,
    }


def _print_metric(tag: str, metric: dict) -> None:
    print(f'[{tag}] mIoU={metric["mIoU"]:.2f} mDice={metric["mDice"]:.2f} '
          f'mAcc={metric["mAcc"]:.2f} aAcc={metric["aAcc"]:.2f}')
    for i, cls in enumerate(OilSpillDataset.CLASSES):
        print(f'[{tag}] {cls}: IoU={metric["IoU"][i]:.2f} '
              f'Dice={metric["Dice"][i]:.2f} Acc={metric["Acc"][i]:.2f}')


def _save_md(path: str, tag: str, metric: dict) -> None:
    lines = [
        f'\n### {tag}\n',
        '| 指标 | 数值 |',
        '|------|------|',
        f'| mIoU | **{metric["mIoU"]:.2f}** |',
        f'| mDice | **{metric["mDice"]:.2f}** |',
        f'| mAcc | **{metric["mAcc"]:.2f}** |',
        f'| aAcc | **{metric["aAcc"]:.2f}** |',
        '',
        '| 类别 | IoU | Dice | Acc |',
        '|------|-----|------|-----|',
    ]
    for i, cls in enumerate(OilSpillDataset.CLASSES):
        lines.append(
            f'| {cls} | {metric["IoU"][i]:.2f} | {metric["Dice"][i]:.2f} | {metric["Acc"][i]:.2f} |'
        )
    lines.append('')
    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Test with 4-flip TTA')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', default='outputs/test_tta')
    parser.add_argument('--slide', action='store_true',
                        help='Use sliding-window inference')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--save-md', default=None,
                        help='Path to markdown file to append results')
    parser.add_argument('--opts', nargs='*')
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    ds = OilSpillDataset(
        root=cfg['data']['root'], split='test', transform=EvalTransform())
    loader = DataLoader(
        ds,
        batch_size=int(cfg['data']['val_batch_size']),
        shuffle=False,
        num_workers=int(cfg['data']['num_workers']),
        pin_memory=True)

    slide_cfg = _build_slide_cfg(cfg, args)
    num_classes = cfg['model']['decode_head']['num_classes']

    tag = 'slide_tta' if slide_cfg is not None else 'whole_tta'
    metric = evaluate_tta(model, loader, device, num_classes, slide_cfg)
    _print_metric(tag, metric)

    if args.save_md:
        _save_md(args.save_md, tag, metric)
        print(f'[save-md] appended to {args.save_md}')


if __name__ == '__main__':
    main()
