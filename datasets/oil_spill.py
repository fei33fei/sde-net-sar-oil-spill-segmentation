import os
from glob import glob
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    CLASSES = ('sea_surface', 'oil_spill', 'look_alike', 'ship', 'land')
    PALETTE = np.array(
        [[0, 0, 0], [0, 255, 255], [255, 0, 0], [153, 76, 0], [0, 153, 0]],
        dtype=np.uint8)

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform=None,
                 with_mask: bool = True) -> None:
        self.root = root
        self.split = split
        self.transform = transform
        self.with_mask = with_mask

        self.img_dir = os.path.join(root, split, 'images')
        self.mask_dir = os.path.join(root, split, 'labels_1D')
        self.images = sorted(glob(os.path.join(self.img_dir, '*.jpg')))
        if len(self.images) == 0:
            raise RuntimeError(f'No images found in {self.img_dir}')

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.images[idx]
        name = os.path.basename(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {'image': image, 'name': name}
        if self.with_mask:
            mask_path = os.path.join(self.mask_dir, name.replace('.jpg', '.png'))
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f'Cannot read mask: {mask_path}')
            sample['mask'] = mask

        if self.transform is not None:
            sample = self.transform(sample)
            sample['name'] = name
        return sample
