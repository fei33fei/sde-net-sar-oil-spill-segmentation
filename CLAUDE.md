# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Standalone PyTorch project for SAR marine oil spill semantic segmentation using ConvNeXt V2 backbone with various UPerNet-style decoders. Supports single GPU and DDP distributed training, with configurable decoder heads for experiments.

## Common Commands

```bash
# Train (single GPU)
python tools/train.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml

# Train (DDP, 2 GPU)
python tools/train.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml --launcher pytorch

# Test with checkpoint
python tools/test.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml --checkpoint work_dirs/.../best_mIoU.pth

# Test with sliding window
python tools/test.py --config configs/xxx.yaml --checkpoint path --slide

# Run unit tests
pytest tests/
```

## Architecture

The codebase is organized into:

- `models/backbone/` — ConvNeXtV2 backbone
- `models/decoder/` — Various decode heads (UPerHead, EnhancedUPerHead, SGFR, DGFPN, OLDM variants)
- `models/modules/` — Reusable modules (LGDA, SPCM, ClassAwareFPN, SGFR, OLDM, etc.)
- `datasets/` — OilSpillDataset and transforms
- `engine/` — Trainer, evaluator, inference
- `utils/` — Config, checkpoint, distributed, visualization
- `tools/` — train.py, test.py, infer.py scripts
- `configs/` — YAML configuration files

### Model Selection

`tools/train.py` selects model based on `decode_head.type` in config:

- `UPerHead` → `EncoderDecoder` (`models/segmentor.py`)
- `EnhancedUPerHead` → `EnhancedEncoderDecoder` (`models/segmentor_enhanced.py`)
- `UPerHeadOLDM` → `OLDMEncoderDecoder` (`models/segmentor_oldm.py`)
- `UPerHeadSGFRDGFPNOLDM` → `SGFRDGFPNOLDMEncoderDecoder` (`models/segmentor_sgfr_dgfpn_oldm.py`)
- Other variants: OLDMPartialCH, OLDMGatedTD, OLDMLgdaSpcm, OLDMSpcm, PSABSMFFOLDM, DGFPNOLDM, SGFR, Ablation

### Current Main Configuration

`configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml` uses:
- Backbone: ConvNeXt V2-B (depths: [3,3,27,3], dims: [128,256,512,1024])
- Decode head: `UPerHeadSGFRDGFPNOLDM` with SGFR on shallow laterals, SG-PPM on C4, DG-FPN, OLDM
- Classes: 5 (sea_surface, oil_spill, look_alike, ship, land)
- Loss: CE + 2.0 * Dice with class weights

### Dataset

`OilSpillDataset` (`datasets/oil_spill.py`) loads SAR oil spill images from directory structure:
- `root/split/` with images and masks
- 5 classes: sea_surface, oil_spill, look_alike, ship, land
- Class names accessed via `OilSpillDataset.CLASSES`

## Key Patterns

- All model classes inherit from `nn.Module`
- Use `build_loss()` from `models/losses.py` to construct loss functions
- Config is loaded via `utils.config.load_config()`
- Checkpoints saved/loaded via `utils/checkpoint.py`
- Distributed training uses `torch.distributed` with `init_distributed()` from `utils/dist.py`