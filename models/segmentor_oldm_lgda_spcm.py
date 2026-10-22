"""Encoder-decoder segmentor with LGDA + SPCM + vanilla FPN + OLDM.

Uses ``UPerHeadOLDMLgdaSpcm`` as the main decode head.  Loss function
is built via ``build_loss`` for fair ablation comparison.
Auxiliary head receives raw backbone ``features[2]`` (C3).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.backbone.convnext_v2 import ConvNeXtV2
from models.decoder.fcn_head import FCNHead
from models.decoder.uper_head_oldm_lgda_spcm import UPerHeadOLDMLgdaSpcm
from models.losses import build_loss


class OLDMLgdaSpcmEncoderDecoder(nn.Module):

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        bcfg = cfg['model']['backbone']
        dcfg = cfg['model']['decode_head']
        acfg = cfg['model']['auxiliary_head']
        lcfg = cfg['model'].get('loss', {})

        self.backbone = ConvNeXtV2(
            in_chans=bcfg.get('in_chans', 3),
            depths=bcfg['depths'],
            dims=bcfg['dims'],
            drop_path_rate=bcfg.get('drop_path_rate', 0.4),
            out_indices=bcfg.get('out_indices', (0, 1, 2, 3)),
            frozen_stages=bcfg.get('frozen_stages', -1),
        )
        self.backbone.init_weights(bcfg.get('pretrained', ''))

        self.decode_head = UPerHeadOLDMLgdaSpcm(
            in_channels=tuple(dcfg['in_channels']),
            channels=dcfg['channels'],
            num_classes=dcfg['num_classes'],
            pool_scales=tuple(dcfg.get('pool_scales', [1, 2, 3, 6])),
            dropout_ratio=dcfg.get('dropout_ratio', 0.1),
            use_oldm=bool(dcfg.get('use_oldm', True)),
            oldm_reduction=int(dcfg.get('oldm_reduction', 16)),
            strip_kernel_size=int(dcfg.get('strip_kernel_size', 7)),
            use_local_branch=bool(dcfg.get('use_local_branch', True)),
            use_diagonal_strips=bool(dcfg.get('use_diagonal_strips', False)),
            lgda_num_groups=int(dcfg.get('lgda_num_groups', 16)),
            lgda_spatial_kernel_size=int(dcfg.get('lgda_spatial_kernel_size', 7)),
            lgda_channel_kernel_size=int(dcfg.get('lgda_channel_kernel_size', 5)),
        )

        self.auxiliary_head = FCNHead(
            in_channels=acfg['in_channels'],
            channels=acfg['channels'],
            num_classes=acfg['num_classes'],
            dropout_ratio=acfg.get('dropout_ratio', 0.1),
        )

        self.loss_fn = build_loss(lcfg, num_classes=dcfg['num_classes'])
        self.aux_weight = float(cfg['model'].get('aux_weight', 0.4))

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        output_size = images.shape[2:]

        main_logits = self.decode_head(features, output_size=output_size)
        aux_logits = self.auxiliary_head(features[2], output_size=output_size)

        if masks is None:
            return {'logits': main_logits}

        loss_main = self.loss_fn(main_logits, masks)
        loss_aux = self.loss_fn(aux_logits, masks)
        loss = loss_main + self.aux_weight * loss_aux

        return {
            'loss': loss,
            'loss_main': loss_main.detach(),
            'loss_aux': loss_aux.detach(),
            'logits': main_logits.detach(),
        }

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward(images)['logits']
        return torch.argmax(logits, dim=1)
