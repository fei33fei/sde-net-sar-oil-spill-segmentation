import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch


def normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


def photometric_distortion(image: np.ndarray,
                           brightness_delta: int = 32,
                           contrast_range: Tuple[float, float] = (0.5, 1.5),
                           saturation_range: Tuple[float, float] = (0.5, 1.5),
                           hue_delta: int = 18) -> np.ndarray:
    """Aligned with mmseg PhotoMetricDistortion."""
    img = image.astype(np.float32)
    # brightness
    if random.randint(0, 1):
        delta = random.uniform(-brightness_delta, brightness_delta)
        img = np.clip(img + delta, 0, 255)
    # contrast mode 0 (before HSV) or mode 1 (after HSV)
    mode = random.randint(0, 1)
    if mode == 0:
        if random.randint(0, 1):
            alpha = random.uniform(*contrast_range)
            img = np.clip(img * alpha, 0, 255)
    # saturation & hue
    img = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    if random.randint(0, 1):
        hsv[..., 1] *= random.uniform(*saturation_range)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    if random.randint(0, 1):
        hsv[..., 0] += random.uniform(-hue_delta, hue_delta)
        hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # contrast mode 1
    if mode == 1:
        if random.randint(0, 1):
            alpha = random.uniform(*contrast_range)
            img = np.clip(img * alpha, 0, 255)
    return img.astype(np.uint8)


def sar_speckle_noise(image: np.ndarray, std: float = 0.08) -> np.ndarray:
    """Multiplicative speckle noise simulating SAR imaging artefacts."""
    noise = np.random.normal(1.0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) * noise, 0, 255).astype(np.uint8)


def sar_gamma(image: np.ndarray, gamma_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Random power-law gamma correction on intensity."""
    gamma = random.uniform(*gamma_range)
    table = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
    return cv2.LUT(image, table.astype(np.uint8))


def sar_contrast(image: np.ndarray, contrast_range: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """Mild global contrast perturbation."""
    alpha = random.uniform(*contrast_range)
    return np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


class TrainTransform:

    def __init__(self, base_scale=(1250, 650), crop_size=(512, 512),
                 cat_max_ratio: float = 0.75, ignore_index: int = 255,
                 photometric_distortion_enabled: bool = True,
                 sar_aug: dict | None = None,
                 whole_resize_prob: float = 0.0):
        self.base_scale = base_scale
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.photometric_distortion_enabled = photometric_distortion_enabled
        self.whole_resize_prob = whole_resize_prob

        sar = sar_aug or {}
        self.sar_aug_enabled = sar.get('enabled', False)
        self.speckle_prob = float(sar.get('speckle_prob', 0.5))
        self.speckle_std = float(sar.get('speckle_std', 0.08))
        self.gamma_prob = float(sar.get('gamma_prob', 0.5))
        self.gamma_range = tuple(sar.get('gamma_range', [0.8, 1.2]))
        self.contrast_prob = float(sar.get('contrast_prob', 0.5))
        self.contrast_range = tuple(sar.get('contrast_range', [0.85, 1.15]))

    def _random_crop(self, image: np.ndarray, mask: np.ndarray):
        """Crop with cat_max_ratio filtering, aligned with mmseg RandomCrop."""
        ch, cw = self.crop_size[1], self.crop_size[0]
        h, w = image.shape[:2]

        if h < ch or w < cw:
            pad_h = max(ch - h, 0)
            pad_w = max(cw - w, 0)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            mask = cv2.copyMakeBorder(
                mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_index)
            h, w = image.shape[:2]

        for _ in range(10):
            y1 = random.randint(0, h - ch)
            x1 = random.randint(0, w - cw)
            crop_mask = mask[y1:y1 + ch, x1:x1 + cw]
            if self.cat_max_ratio < 1.0:
                labels, cnt = np.unique(crop_mask, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
            else:
                break

        image = image[y1:y1 + ch, x1:x1 + cw]
        mask = mask[y1:y1 + ch, x1:x1 + cw]
        return image, mask

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        image = sample['image']
        mask = sample['mask']

        if self.whole_resize_prob > 0 and random.random() < self.whole_resize_prob:
            image = cv2.resize(
                image, (self.crop_size[0], self.crop_size[1]),
                interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(
                mask, (self.crop_size[0], self.crop_size[1]),
                interpolation=cv2.INTER_NEAREST)
        else:
            # RandomResize
            ratio = random.uniform(0.5, 2.0)
            target_w = int(self.base_scale[0] * ratio)
            target_h = int(self.base_scale[1] * ratio)
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            # RandomCrop with cat_max_ratio
            image, mask = self._random_crop(image, mask)

        # RandomFlip — horizontal
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # RandomFlip — vertical
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Random 90°/180°/270° rotation (k=0 means no rotation)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # Intensity-level augmentations (mutually exclusive by design)
        if self.sar_aug_enabled:
            if random.random() < self.speckle_prob:
                image = sar_speckle_noise(image, std=self.speckle_std)
            if random.random() < self.gamma_prob:
                image = sar_gamma(image, gamma_range=self.gamma_range)
            if random.random() < self.contrast_prob:
                image = sar_contrast(image, contrast_range=self.contrast_range)
        elif self.photometric_distortion_enabled:
            image = photometric_distortion(image)

        image = normalize(image)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.astype(np.int64))
        return {'image': image, 'mask': mask}


class EvalTransform:

    def __init__(self, resize=(1250, 650)):
        self.resize = resize

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        image = sample['image']
        mask = sample.get('mask')

        image = cv2.resize(
            image, self.resize, interpolation=cv2.INTER_LINEAR)
        image = normalize(image)
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()

        out = {'image': image_t}
        if mask is not None:
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
            out['mask'] = torch.from_numpy(mask.astype(np.int64))
        return out
