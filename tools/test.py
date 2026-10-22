import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.oil_spill import OilSpillDataset
from datasets.transforms import EvalTransform
from engine.evaluator import evaluate_miou
from tools.train import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def _save_md(path: str, tag: str, metric: dict) -> None:
    """Append evaluation results as a markdown table to *path*."""
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


def _build_slide_cfg(cfg, args):
    """Build slide_cfg dict from CLI flags and yaml config.

    Priority: CLI ``--slide`` flag enables sliding window; ``--overlap``
    overrides the yaml value.  Returns *None* when whole-image mode is used.
    """
    test_cfg = cfg.get('test', {})
    mode = test_cfg.get('mode', 'whole')
    if args.slide:
        mode = 'slide'
    if mode != 'slide':
        return None

    crop_size = list(test_cfg.get('crop_size', cfg['data']['crop_size']))
    overlap = args.overlap if args.overlap is not None else float(test_cfg.get('overlap', 0.2))
    stride = [int(c * (1 - overlap)) for c in crop_size]
    print(f'[slide] crop_size={crop_size}, stride={stride}, overlap={overlap}')
    return {'crop_size': crop_size, 'stride': stride}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', default='outputs/test')
    parser.add_argument('--slide', action='store_true',
                        help='Use sliding-window inference')
    parser.add_argument('--overlap', type=float, default=None,
                        help='Overlap ratio for sliding window (default: 0.2)')
    parser.add_argument(
        '--slide-aggregate',
        choices=('mean_logits', 'vote', 'both'),
        default=None,
        help='Slide fusion: averaged logits, majority vote, or run both (default: both when slide)')
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

    def _print_metric(tag: str, metric: dict) -> None:
        print(f'[{tag}] mIoU={metric["mIoU"]:.2f} mDice={metric["mDice"]:.2f} '
              f'mAcc={metric["mAcc"]:.2f} aAcc={metric["aAcc"]:.2f}')
        for i, cls in enumerate(OilSpillDataset.CLASSES):
            print(f'[{tag}] {cls}: IoU={metric["IoU"][i]:.2f} '
                  f'Dice={metric["Dice"][i]:.2f} Acc={metric["Acc"][i]:.2f}')

    if slide_cfg is None:
        metric = evaluate_miou(
            model, loader, device, num_classes, slide_cfg=None)
        _print_metric('whole', metric)
        if args.save_md:
            _save_md(args.save_md, 'whole', metric)
    else:
        agg_arg = args.slide_aggregate
        if agg_arg is None:
            agg_arg = cfg.get('test', {}).get('slide_aggregate', 'both')
        aggs = ['mean_logits', 'vote'] if agg_arg == 'both' else [agg_arg]
        base = {k: v for k, v in slide_cfg.items() if k != 'aggregate'}
        for agg in aggs:
            sc = {**base, 'aggregate': agg}
            print(f'[slide] aggregate={agg}')
            metric = evaluate_miou(
                model, loader, device, num_classes, slide_cfg=sc)
            tag = f'slide/{agg}'
            _print_metric(tag, metric)
            if args.save_md:
                _save_md(args.save_md, tag, metric)

    if args.save_md:
        print(f'[save-md] appended to {args.save_md}')


if __name__ == '__main__':
    main()
