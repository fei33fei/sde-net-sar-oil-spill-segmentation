from collections import OrderedDict
from typing import Sequence, Tuple

import torch
import torch.nn as nn

from models.modules.layers import DropPath, GRN, LayerNorm


class ConvNeXtV2Block(nn.Module):

    def __init__(self, dim: int, drop_path: float = 0.) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_last')
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return identity + self.drop_path(x)


class ConvNeXtV2(nn.Module):

    def __init__(self,
                 in_chans: int = 3,
                 depths: Sequence[int] = (3, 3, 9, 3),
                 dims: Sequence[int] = (96, 192, 384, 768),
                 drop_path_rate: float = 0.4,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 frozen_stages: int = -1) -> None:
        super().__init__()
        self.depths = list(depths)
        self.dims = list(dims)
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norms = nn.ModuleList()
        for i in range(4):
            self.norms.append(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'))

    def init_weights(self, checkpoint: str = '') -> None:
        if not checkpoint:
            return
        import os
        if os.path.isfile(checkpoint):
            ckpt = torch.load(checkpoint, map_location='cpu')
        else:
            ckpt = torch.hub.load_state_dict_from_url(
                checkpoint, map_location='cpu', check_hash=False)
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('backbone.'):
                k = k[9:]
            if k.startswith('head.') or k.startswith('norm.'):
                continue
            clean_state_dict[k] = v
        self.load_state_dict(clean_state_dict, strict=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        return self

    def _freeze_stages(self) -> None:
        if self.frozen_stages < 0:
            return
        for i in range(self.frozen_stages + 1):
            self.downsample_layers[i].eval()
            self.stages[i].eval()
            self.norms[i].eval()
            for p in self.downsample_layers[i].parameters():
                p.requires_grad = False
            for p in self.stages[i].parameters():
                p.requires_grad = False
            for p in self.norms[i].parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(self.norms[i](x))
        return tuple(outs)
