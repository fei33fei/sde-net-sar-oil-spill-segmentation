import copy
import logging
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from engine.evaluator import evaluate_miou
from engine.optimizer import build_optimizer, build_scheduler
from utils.checkpoint import save_checkpoint
from utils.dist import get_rank, is_dist_avail_and_initialized, reduce_tensor


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    Uses a warmup ramp: effective_decay = min(decay, (1+step)/(10+step))
    so early steps track the live model closely (low decay) and the decay
    gradually ramps up to the target value.  This avoids the cold-start
    problem where EMA is stuck near random initialisation for small datasets.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.target_decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.num_updates += 1
        # Ramp: starts near 0.1, reaches ~0.999 after ~9000 steps.
        decay = min(self.target_decay,
                    (1 + self.num_updates) / (10 + self.num_updates))
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(decay).add_(m_param.data, alpha=1 - decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_val_slide_cfg(cfg):
    """Return slide_cfg for validation if ``train.val_slide`` is enabled."""
    if not cfg.get('train', {}).get('val_slide', False):
        return None
    test_cfg = cfg.get('test', {})
    crop_size = list(test_cfg.get('crop_size', cfg['data']['crop_size']))
    overlap = float(test_cfg.get('overlap', 0.2))
    stride = [int(c * (1 - overlap)) for c in crop_size]
    return {'crop_size': crop_size, 'stride': stride}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, cfg, device, args):
    logger = logging.getLogger('train')
    model = model.to(device)

    # EMA — operates on the unwrapped model, created before DDP wrapping.
    ema_cfg = cfg.get('train', {}).get('ema', {})
    ema_enabled = ema_cfg.get('enabled', False)
    ema: ModelEMA | None = None
    if ema_enabled:
        ema = ModelEMA(model, decay=float(ema_cfg.get('decay', 0.9999)))
        logger.info(f'EMA enabled (target_decay={ema.target_decay}, warmup ramp)')

    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Gradient clipping
    grad_clip = float(cfg.get('train', {}).get('grad_clip', 0.0))
    if grad_clip > 0:
        logger.info(f'Gradient clipping max_norm={grad_clip}')

    max_epochs = int(cfg['train']['epochs'])
    val_interval = int(cfg['train']['val_interval'])
    ckpt_interval = int(cfg['train']['checkpoint_interval'])
    work_dir = cfg['train']['work_dir']
    os.makedirs(work_dir, exist_ok=True)

    best_miou = -1.0
    best_path = os.path.join(work_dir, 'best_mIoU.pth')

    for epoch in range(max_epochs):
        model.train()
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        tic = time.time()
        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(images, masks)
                loss = out['loss']

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.detach()

            if ema is not None:
                raw = model.module if isinstance(model, DDP) else model
                ema.update(raw)

        running_loss = running_loss / len(train_loader)
        if is_dist_avail_and_initialized():
            running_loss = reduce_tensor(running_loss, average=True)
        scheduler.step()

        if get_rank() == 0:
            logger.info(
                f'Epoch [{epoch + 1}/{max_epochs}] loss={running_loss.item():.4f} '
                f"lr={optimizer.param_groups[0]['lr']:.6e} "
                f'time={time.time() - tic:.1f}s')

        need_val = (epoch + 1) % val_interval == 0 or (epoch + 1) == max_epochs
        if need_val:
            num_classes = cfg['model']['decode_head']['num_classes']
            slide_cfg = _build_val_slide_cfg(cfg)

            # Validate with EMA weights when available.
            val_model = ema.shadow if ema is not None else (
                model.module if isinstance(model, DDP) else model)
            metric = evaluate_miou(
                val_model, val_loader, device, num_classes,
                slide_cfg=slide_cfg)
            if get_rank() == 0:
                ema_tag = ' (EMA)' if ema is not None else ''
                logger.info(
                    f"Val epoch={epoch + 1}{ema_tag} mIoU={metric['mIoU']:.2f} "
                    f"aAcc={metric['aAcc']:.2f} mAcc={metric['mAcc']:.2f}")
                class_names = cfg.get('class_names', None)
                iou_parts = []
                for c in range(num_classes):
                    name = class_names[c] if class_names and c < len(class_names) else f'cls{c}'
                    iou_parts.append(f'{name}={metric["IoU"][c]:.2f}')
                logger.info(f'  Per-class IoU: {", ".join(iou_parts)}')
                if metric['mIoU'] > best_miou:
                    best_miou = metric['mIoU']
                    save_model = ema.shadow if ema is not None else (
                        model.module if isinstance(model, DDP) else model)
                    save_checkpoint(
                        save_model,
                        optimizer, scheduler, epoch + 1, best_path, best_miou)
                    logger.info(f'New best mIoU {best_miou:.2f}, saved {best_path}')

        if get_rank() == 0 and ((epoch + 1) % ckpt_interval == 0):
            path = os.path.join(work_dir, f'epoch_{epoch + 1}.pth')
            save_model = ema.shadow if ema is not None else (
                model.module if isinstance(model, DDP) else model)
            save_checkpoint(
                save_model,
                optimizer, scheduler, epoch + 1, path, best_miou)

    if get_rank() == 0:
        logger.info(f'Training done. Best mIoU={best_miou:.2f}')
