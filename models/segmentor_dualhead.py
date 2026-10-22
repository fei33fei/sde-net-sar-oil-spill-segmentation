"""Dual-head enhanced segmentor: 5-class main + oil/look_alike binary head.

Extends the ``EnhancedEncoderDecoder`` architecture with a lightweight binary
classification head that distinguishes oil_spill from look_alike.  The binary
head receives the same ``enhanced_c3`` feature as the auxiliary head.

Training:
    loss = loss_main + aux_weight * loss_aux + binary_weight * loss_binary

    ``loss_binary`` is a masked CE computed **only** on pixels whose GT label is
    ``oil_class_id`` or ``la_class_id``; all other pixels are ignored.

Inference:
    The binary head's logit difference is used to shift the main logits for the
    oil and look_alike channels in logit space, which is compatible with
    ``slide_inference``'s ``mean_logits`` accumulation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.convnext_v2 import ConvNeXtV2
from models.decoder.enhanced_uper_head import EnhancedUPerHead
from models.decoder.fcn_head import FCNHead
from models.losses import build_loss


class DualHeadEnhancedEncoderDecoder(nn.Module):

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        bcfg = cfg['model']['backbone']
        dcfg = cfg['model']['decode_head']
        acfg = cfg['model']['auxiliary_head']
        lcfg = cfg['model'].get('loss', {})
        bincfg = cfg['model']['binary_head']

        self.backbone = ConvNeXtV2(
            in_chans=bcfg.get('in_chans', 3),
            depths=bcfg['depths'],
            dims=bcfg['dims'],
            drop_path_rate=bcfg.get('drop_path_rate', 0.4),
            out_indices=bcfg.get('out_indices', (0, 1, 2, 3)),
            frozen_stages=bcfg.get('frozen_stages', -1))
        self.backbone.init_weights(bcfg.get('pretrained', ''))

        proto_contrast_cfg = dcfg.get('proto_contrast', None)
        self.decode_head = EnhancedUPerHead(
            in_channels=tuple(dcfg['in_channels']),
            channels=dcfg['channels'],
            num_classes=dcfg['num_classes'],
            pool_scales=tuple(dcfg.get('pool_scales', [1, 2, 3, 6])),
            strip_kernel_size=int(dcfg.get('strip_kernel_size', 7)),
            use_local_branch=bool(dcfg.get('use_local_branch', True)),
            use_diagonal_strips=bool(dcfg.get('use_diagonal_strips', False)),
            lgda_num_groups=int(dcfg.get('lgda_num_groups', 16)),
            lgda_spatial_kernel_size=int(dcfg.get('lgda_spatial_kernel_size', 7)),
            lgda_channel_kernel_size=int(dcfg.get('lgda_channel_kernel_size', 5)),
            dropout_ratio=dcfg.get('dropout_ratio', 0.1),
            proto_contrast=proto_contrast_cfg,
        )
        self._proto_inter_weight = float(
            (proto_contrast_cfg or {}).get('inter_weight', 0.1)
        )
        self._proto_align_weight = float(
            (proto_contrast_cfg or {}).get('align_weight', 0.1)
        )

        self.auxiliary_head = FCNHead(
            in_channels=acfg['in_channels'],
            channels=acfg['channels'],
            num_classes=acfg['num_classes'],
            dropout_ratio=acfg.get('dropout_ratio', 0.1))

        self.binary_head = FCNHead(
            in_channels=int(bincfg['in_channels']),
            channels=int(bincfg['channels']),
            num_classes=2,
            dropout_ratio=bincfg.get('dropout_ratio', 0.1))

        self.loss_fn = build_loss(lcfg, num_classes=dcfg['num_classes'])
        self.aux_weight = float(cfg['model'].get('aux_weight', 0.4))
        self.binary_weight = float(cfg['model'].get('binary_weight', 0.5))
        self.binary_refine_weight = float(
            cfg['model'].get('binary_refine_weight', 0.3)
        )
        self.oil_class_id = int(cfg['model'].get('oil_class_id', 1))
        self.la_class_id = int(cfg['model'].get('la_class_id', 2))

    def _binary_loss(
        self, binary_logits: torch.Tensor, masks: torch.Tensor,
    ) -> torch.Tensor:
        binary_target = masks.clone().long()
        oil_mask = masks == self.oil_class_id
        la_mask = masks == self.la_class_id
        binary_target[oil_mask] = 0
        binary_target[la_mask] = 1
        binary_target[~(oil_mask | la_mask)] = 255
        return F.cross_entropy(binary_logits, binary_target, ignore_index=255)

    def _refine_logits(
        self, main_logits: torch.Tensor, binary_logits: torch.Tensor,
    ) -> torch.Tensor:
        delta = binary_logits[:, 0] - binary_logits[:, 1]
        w = self.binary_refine_weight
        refined = main_logits.clone()
        refined[:, self.oil_class_id] = refined[:, self.oil_class_id] + w * delta
        refined[:, self.la_class_id] = refined[:, self.la_class_id] - w * delta
        return refined

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        output_size = images.shape[2:]

        head_out = self.decode_head(
            features, output_size=output_size, masks=masks,
        )

        proto_losses: Optional[Dict[str, torch.Tensor]] = None
        if len(head_out) == 3:
            main_logits, enhanced_c3, proto_losses = head_out
        else:
            main_logits, enhanced_c3 = head_out

        binary_logits = self.binary_head(enhanced_c3, output_size=output_size)

        if masks is None:
            return {'logits': self._refine_logits(main_logits, binary_logits)}

        aux_logits = self.auxiliary_head(enhanced_c3, output_size=output_size)

        loss_main = self.loss_fn(main_logits, masks)
        loss_aux = self.loss_fn(aux_logits, masks)
        loss_binary = self._binary_loss(binary_logits, masks)
        loss = (
            loss_main
            + self.aux_weight * loss_aux
            + self.binary_weight * loss_binary
        )

        result: Dict[str, torch.Tensor] = {
            'loss_main': loss_main.detach(),
            'loss_aux': loss_aux.detach(),
            'loss_binary': loss_binary.detach(),
            'logits': main_logits.detach(),
        }

        if proto_losses is not None:
            l_inter = proto_losses['loss_inter_proto']
            l_align = proto_losses['loss_pixel_align']
            loss = loss + (
                self._proto_inter_weight * l_inter
                + self._proto_align_weight * l_align
            )
            result['loss_inter_proto'] = l_inter.detach()
            result['loss_pixel_align'] = l_align.detach()

        result['loss'] = loss
        return result

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward(images)['logits']
        return torch.argmax(logits, dim=1)
