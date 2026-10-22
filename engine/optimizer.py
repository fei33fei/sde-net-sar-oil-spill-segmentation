import logging

import torch


def _get_convnextv2_layer_id(param_name: str, depths: list[int]) -> int:
    """Map a parameter name to its depth-layer id for ConvNeXtV2 LLRD.

    Deeper layers → higher id → higher LR.  Non-backbone parameters
    (decode head, aux head) receive the max id so they train at full LR.
    """
    num_layers = sum(depths) + len(depths)

    name = param_name.removeprefix('module.')

    if not name.startswith('backbone.'):
        return num_layers

    name = name.removeprefix('backbone.')

    if name.startswith('downsample_layers'):
        stage_id = int(name.split('.')[1])
        if stage_id == 0:
            return 0
        return sum(depths[:stage_id]) + stage_id

    if name.startswith('stages'):
        stage_id = int(name.split('.')[1])
        block_id = int(name.split('.')[2])
        return sum(depths[:stage_id]) + stage_id + 1 + block_id

    if name.startswith('norms'):
        stage_id = int(name.split('.')[1])
        return sum(depths[:stage_id]) + stage_id + depths[stage_id]

    return num_layers


def _no_weight_decay(param_name: str) -> bool:
    """Norm parameters, all biases and GRN gamma/beta skip weight decay."""
    if param_name.endswith('.bias'):
        return True
    if 'norm' in param_name:
        return True
    if 'grn.' in param_name:
        return True
    return False


def build_optimizer(model, cfg: dict):
    ocfg = cfg['train']['optimizer']
    assert ocfg['type'] == 'AdamW'

    base_lr = float(ocfg['lr'])
    weight_decay = float(ocfg['weight_decay'])
    layer_decay_rate = float(ocfg.get('layer_decay_rate', 1.0))

    if layer_decay_rate >= 1.0:
        return torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay)

    depths = list(cfg['model']['backbone']['depths'])
    num_layers = sum(depths) + len(depths)
    logger = logging.getLogger('train')

    group_map: dict[tuple[int, bool], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = _get_convnextv2_layer_id(name, depths)
        lr_scale = layer_decay_rate ** (num_layers - layer_id)
        no_wd = _no_weight_decay(name)

        key = (layer_id, no_wd)
        if key not in group_map:
            group_map[key] = {
                'params': [],
                'lr': base_lr * lr_scale,
                'weight_decay': 0.0 if no_wd else weight_decay,
                'lr_scale': lr_scale,
                'layer_id': layer_id,
            }
        group_map[key]['params'].append(param)

    groups = sorted(group_map.values(), key=lambda g: (g['layer_id'], g['weight_decay']))
    logger.info(
        f'LLRD: {len(groups)} param groups, decay_rate={layer_decay_rate}, '
        f'num_layers={num_layers}, lr_scale range '
        f'[{min(g["lr_scale"] for g in groups):.4f}, '
        f'{max(g["lr_scale"] for g in groups):.4f}]')

    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def build_scheduler(optimizer, cfg: dict):
    max_epochs = int(cfg['train']['epochs'])
    warmup_epochs = int(cfg['train']['scheduler']['warmup_epochs'])
    power = float(cfg['train']['scheduler'].get('power', 0.9))
    start_factor = float(cfg['train']['scheduler'].get('start_factor', 1e-6))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Aligned with mmseg LinearLR: epoch 0 → start_factor, epoch warmup_epochs → 1.0
            alpha = epoch / max(1, warmup_epochs)
            return start_factor + (1.0 - start_factor) * alpha
        t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return (1 - t)**power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
