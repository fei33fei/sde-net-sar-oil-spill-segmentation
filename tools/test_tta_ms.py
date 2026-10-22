"""Test script with 4-flip + 4-scale (8x) TTA + sliding window inference.

Usage mirrors ``tools/test_tta.py`` but uses 8 augmentation passes:
  4 flip variants at original scale  +  4 scale variants without flip.

Example:
    python tools/test_tta_ms.py \
        --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml \
        --checkpoint work_dirs/.../best_mIoU.pth \
        --slide \
        --save-md work_dirs/.../测试结果.md
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from datasets.oil_spill import OilSpillDataset
from datasets.transforms import EvalTransform
from engine.evaluator import intersect_and_union
from engine.inference_tta_ms import slide_inference_tta_ms
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
    overlap = (args.overlap if args.overlap is not None
               else float(test_cfg.get('overlap', 0.2)))
    stride = [int(c * (1 - overlap)) for c in crop_size]
    print(f'[slide-tta-ms] crop_size={crop_size}, stride={stride}, '
          f'overlap={overlap}')
    return {'crop_size': crop_size, 'stride': stride}


def _parse_scales(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in raw.split(','))


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes, slide_cfg, scales):
    model.eval()
    total_inter = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_label = torch.zeros(num_classes, device=device)
    total_correct = torch.zeros(1, device=device)
    total_pixel = torch.zeros(1, device=device)

    n_samples = len(dataloader)
    t0 = time.time()

    for idx, batch in enumerate(dataloader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        if slide_cfg is not None:
            logits = slide_inference_tta_ms(
                model, images,
                crop_size=tuple(slide_cfg['crop_size']),
                stride=tuple(slide_cfg['stride']),
                num_classes=num_classes,
                scales=scales,
            )
        else:
            logits = slide_inference_tta_ms(
                model, images,
                crop_size=tuple(images.shape[2:]),
                stride=tuple(images.shape[2:]),
                num_classes=num_classes,
                scales=scales,
            )

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

        if (idx + 1) % 10 == 0 or (idx + 1) == n_samples:
            elapsed = time.time() - t0
            print(f'  [{idx + 1}/{n_samples}] {elapsed:.1f}s')

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
            f'| {cls} | {metric["IoU"][i]:.2f} '
            f'| {metric["Dice"][i]:.2f} | {metric["Acc"][i]:.2f} |'
        )
    lines.append('')
    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Test with 4-flip + 4-scale (8x) TTA')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', default='outputs/test_tta_ms')
    parser.add_argument('--slide', action='store_true',
                        help='Use sliding-window inference')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--scales', type=str, default='1,1.25,1.5,1.75',
                        help='Comma-separated scale factors (default: 1,1.25,1.5,1.75)')
    parser.add_argument('--save-md', default=None,
                        help='Path to markdown file to append results')
    parser.add_argument('--opts', nargs='*')
    args = parser.parse_args()

    scales = _parse_scales(args.scales)
    print(f'[tta-ms] scales={scales}, n_passes={4 + len(scales)}')

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

    tag = 'slide_tta_ms' if slide_cfg is not None else 'whole_tta_ms'
    metric = evaluate(model, loader, device, num_classes, slide_cfg, scales)
    _print_metric(tag, metric)

    if args.save_md:
        _save_md(args.save_md, tag, metric)
        print(f'[save-md] appended to {args.save_md}')


if __name__ == '__main__':
    main()
