import argparse
import os
import sys
from glob import glob

import cv2
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.segmentor import EncoderDecoder
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.visualize import save_visualization


def preprocess(image_rgb):
    image = cv2.resize(image_rgb, (1250, 650), interpolation=cv2.INTER_LINEAR)
    image = image.astype('float32') / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    image = (torch.from_numpy(image) - mean) / std
    image = image.permute(2, 0, 1).unsqueeze(0).float()
    return image


def list_images(path: str):
    if os.path.isdir(path):
        imgs = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif'):
            imgs.extend(glob(os.path.join(path, ext)))
        return sorted(imgs)
    return [path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--opts', nargs='*')
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncoderDecoder(cfg).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    os.makedirs(args.output, exist_ok=True)
    paths = list_images(args.input)
    with torch.no_grad():
        for p in paths:
            image_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inp = preprocess(image_rgb).to(device)
            logits = model(inp)['logits']
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype('uint8')
            pred = cv2.resize(
                pred,
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            out_path = os.path.join(args.output, os.path.basename(p))
            save_visualization(image_rgb, pred, out_path)
            print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
