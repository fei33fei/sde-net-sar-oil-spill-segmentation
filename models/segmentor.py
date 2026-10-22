from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.convnext_v2 import ConvNeXtV2
from models.decoder.fcn_head import FCNHead
from models.decoder.uper_head import UPerHead


class EncoderDecoder(nn.Module):

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        bcfg = cfg['model']['backbone']
        dcfg = cfg['model']['decode_head']
        acfg = cfg['model']['auxiliary_head']

        self.backbone = ConvNeXtV2(
            in_chans=bcfg.get('in_chans', 3),
            depths=bcfg['depths'],
            dims=bcfg['dims'],
            drop_path_rate=bcfg.get('drop_path_rate', 0.4),
            out_indices=bcfg.get('out_indices', (0, 1, 2, 3)),
            frozen_stages=bcfg.get('frozen_stages', -1))
        self.backbone.init_weights(bcfg.get('pretrained', ''))

        self.decode_head = UPerHead(
            in_channels=tuple(dcfg['in_channels']),
            channels=dcfg['channels'],
            num_classes=dcfg['num_classes'],
            pool_scales=tuple(dcfg.get('pool_scales', [1, 2, 3, 6])),
            dropout_ratio=dcfg.get('dropout_ratio', 0.1))
        self.auxiliary_head = FCNHead(
            in_channels=acfg['in_channels'],
            channels=acfg['channels'],
            num_classes=acfg['num_classes'],
            dropout_ratio=acfg.get('dropout_ratio', 0.1))

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self,
                images: torch.Tensor,
                masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        output_size = images.shape[2:]
        main_logits = self.decode_head(features, output_size=output_size)
        aux_logits = self.auxiliary_head(features[2], output_size=output_size)

        if masks is None:
            return {'logits': main_logits}

        loss_main = self.loss_fn(main_logits, masks.long())
        loss_aux = self.loss_fn(aux_logits, masks.long())
        loss = loss_main + 0.4 * loss_aux
        return {
            'loss': loss,
            'loss_main': loss_main.detach(),
            'loss_aux': loss_aux.detach(),
            'logits': main_logits.detach()
        }

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward(images)['logits']
        return torch.argmax(logits, dim=1)
