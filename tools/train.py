import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, DistributedSampler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.oil_spill import OilSpillDataset
from datasets.transforms import EvalTransform, TrainTransform
from engine.trainer import train_model
from models.segmentor import EncoderDecoder
from models.segmentor_enhanced import EnhancedEncoderDecoder
from utils.config import load_config
from utils.dist import get_rank, init_distributed


def build_model(cfg):
    head_type = cfg['model']['decode_head'].get('type', 'UPerHead')
    if head_type == 'EnhancedUPerHead':
        if 'binary_head' in cfg['model']:
            from models.segmentor_dualhead import DualHeadEnhancedEncoderDecoder
            return DualHeadEnhancedEncoderDecoder(cfg)
        return EnhancedEncoderDecoder(cfg)
    if head_type == 'UPerHeadOLDM':
        from models.segmentor_oldm import OLDMEncoderDecoder
        return OLDMEncoderDecoder(cfg)
    if head_type == 'LiteGFPNOLDMHead':
        from models.segmentor_lite_gfpn_oldm import LiteGFPNOLDMEncoderDecoder
        return LiteGFPNOLDMEncoderDecoder(cfg)
    if head_type == 'UPerHeadOLDMPartialCH':
        from models.segmentor_oldm_partial_ch import OLDMPartialCHEncoderDecoder
        return OLDMPartialCHEncoderDecoder(cfg)
    if head_type == 'UPerHeadOLDMGatedTopDown':
        from models.segmentor_oldm_gated_td import OLDMGatedTDEncoderDecoder
        return OLDMGatedTDEncoderDecoder(cfg)
    if head_type == 'UPerHeadOLDMLgdaSpcm':
        from models.segmentor_oldm_lgda_spcm import OLDMLgdaSpcmEncoderDecoder
        return OLDMLgdaSpcmEncoderDecoder(cfg)
    if head_type == 'UPerHeadOLDMSpcm':
        from models.segmentor_oldm_spcm import OLDMSpcmEncoderDecoder
        return OLDMSpcmEncoderDecoder(cfg)
    if head_type == 'UPerHeadPSABSMFFOLDM':
        from models.segmentor_psab_smff_oldm import PSABSMFFOLDMEncoderDecoder
        return PSABSMFFOLDMEncoderDecoder(cfg)
    if head_type == 'UPerHeadDGFPNOLDM':
        from models.segmentor_dg_fpn_oldm import DGFPNOLDMEncoderDecoder
        return DGFPNOLDMEncoderDecoder(cfg)
    if head_type == 'UPerHeadSGFRDGFPNOLDM':
        from models.segmentor_sgfr_dgfpn_oldm import SGFRDGFPNOLDMEncoderDecoder
        return SGFRDGFPNOLDMEncoderDecoder(cfg)
    if head_type == 'UPerHeadAblation':
        from models.segmentor_ablation import AblationEncoderDecoder
        return AblationEncoderDecoder(cfg)
    if head_type == 'UPerHeadSGFR':
        from models.segmentor_sgfr import SGFREncoderDecoder
        return SGFREncoderDecoder(cfg)
    return EncoderDecoder(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', default='')
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch'])
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--opts', nargs='*')
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    return args


def build_dataloader(cfg, distributed: bool):
    transform_cfg = cfg['data'].get('train_transform', {})
    train_transform = TrainTransform(
        crop_size=tuple(cfg['data'].get('crop_size', [512, 512])),
        **transform_cfg,
    )
    train_ds = OilSpillDataset(
        root=cfg['data']['root'], split='train', transform=train_transform)
    val_ds = OilSpillDataset(
        root=cfg['data']['root'], split='test', transform=EvalTransform())

    train_sampler = DistributedSampler(
        train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(
        val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg['data']['train_batch_size']),
        num_workers=int(cfg['data']['num_workers']),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg['data']['val_batch_size']),
        num_workers=int(cfg['data']['num_workers']),
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True)
    return train_loader, val_loader


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    if args.work_dir:
        cfg['train']['work_dir'] = args.work_dir
    os.makedirs(cfg['train']['work_dir'], exist_ok=True)

    distributed = init_distributed(args.launcher, args.local_rank)
    device = torch.device(
        f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(cfg['train']['work_dir'], 'train.log'),
                mode='a',
                encoding='utf-8')
        ])
    if get_rank() != 0:
        logging.getLogger().setLevel(logging.ERROR)

    model = build_model(cfg)
    train_loader, val_loader = build_dataloader(cfg, distributed)
    train_model(model, train_loader, val_loader, cfg, device, args)


if __name__ == '__main__':
    main()
