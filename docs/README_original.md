# ConvNeXtV2-UPerNet-OilSpill

Standalone PyTorch project for marine oil spill semantic segmentation using
ConvNeXt V2-T backbone and UPerNet decoder.

## Features

- Modular codebase (backbone, decoder, dataset, engine, utils)
- Training with single GPU or DDP
- Validation/testing with mIoU and per-class IoU
- Inference visualization with colored mask overlays
- Enhanced SAR oil-spill model with LGDA, SPCM, ClassAwareGatedFPN, and CE+Dice loss

## Quick Start

1. Install deps:

```bash
pip install -r requirements.txt
```

2. Train:

```bash
python tools/train.py --config configs/convnextv2_t_upernet_oilspill.yaml --launcher pytorch
```

3. Test:

```bash
python tools/test.py --config configs/convnextv2_t_upernet_oilspill.yaml --checkpoint work_dirs/convnextv2_t_upernet_oilspill/best_mIoU.pth
```

4. Infer:

```bash
python tools/infer.py --config configs/convnextv2_t_upernet_oilspill.yaml --checkpoint work_dirs/convnextv2_t_upernet_oilspill/best_mIoU.pth --input /path/to/image_or_dir --output outputs/infer
```

## Enhanced Model

The enhanced SAR oil-spill model is documented in
[`ENHANCED_MODEL.md`](ENHANCED_MODEL.md). It keeps the original UPerNet path
available while adding an independent `EnhancedEncoderDecoder` that uses:

- LGDA on shallow C1/C2/C3 laterals.
- SPCM instead of the original PPM on C4.
- ClassAwareGatedFPN instead of the original top-down FPN.
- CE+Dice loss with class weights for oil-spill and small-object classes.

Train the enhanced model with:

```bash
bash tools/train_enhanced_tmux_2gpu.sh
```
