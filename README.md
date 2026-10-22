# SDE-Net — SAR Marine Oil Spill Semantic Segmentation

**SDE-Net** is an improved encoder–decoder network for pixel-level semantic
segmentation of **oil spills in SAR imagery**. It is built on a ConvNeXt V2
backbone and a UPerNet-style decoder that is redesigned for the three failure
modes that dominate SAR oil-spill segmentation: **speckle-corrupted shallow
texture**, **elongated / fragmented oil-film morphology**, and the
**oil-vs-look-alike ambiguity**.

Everything in this repository is a standalone PyTorch implementation — no
MMEngine / MMCV / MMSegmentation dependency. Only `torch`, `torchvision`,
`pyyaml`, `numpy` and `opencv-python` are required.

---

## Highlights

| Module | Where it sits | What it solves |
|---|---|---|
| **SGFR** — Strip-Guided Feature Refinement | shallow laterals `C1/C2/C3` | Speckle noise and weak boundaries in shallow SAR texture |
| **SG-PPM** — Scale-Gated Pyramid Pooling | deep feature `C4` | Which context scale actually matters for this scene |
| **DG-FPN** — DySample-Gated FPN | top-down fusion path | Detail loss from plain bilinear upsampling; naive element-wise addition |
| **OLDM** — Oil–Look-alike Disambiguation | after fusion, before classifier | `oil_spill` and `look_alike` share near-identical dark-patch signatures |
| **CE + Dice** with class weighting | loss | Small / hard classes (`oil_spill`, `ship`) are swamped by sea background |

### Data flow of the SDE-Net decode head

```
Input SAR image
      │
      ▼
ConvNeXt V2-B backbone            →  C1 (128)  C2 (256)  C3 (512)  C4 (1024)
      │              │              │              │
   1×1 lateral     1×1 lateral     1×1 lateral     │
      ▼              ▼              ▼              │
    SGFR            SGFR           SGFR          SG-PPM  (scale-gated PPM)
      └──────────────┴──────────────┴──────────────┘
                              │
                             DG-FPN  (DySample upsample + sigmoid fusion gates)
                              │
                    concat → FPN bottleneck → dropout
                              │
                            OLDM   (oil / look-alike disambiguation)
                              │
                            cls_seg  →  5-class logits
```

---

## Results

### 1. Ablation study (validation set)

Improvement 1 = ConvNeXt V2 backbone · 2 = SGFR · 3 = SG-PPM ·
4 = DG-FPN + OLDM. `A1` is the ConvNeXt V2 baseline.

**Plain inference (`slide` / `mean_logits`)**

| ID | Configuration | mIoU | mDice | mAcc | aAcc | oil_spill | look_alike | ship |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| A1 | ConvNeXtV2 | 70.46 | 80.94 | 79.63 | 96.55 | 55.94 | 58.89 | 46.44 |
| A2 | +SGFR | 69.32 | 79.90 | 79.47 | 96.11 | 55.90 | 53.16 | 45.19 |
| A3 | +SG-PPM | 69.75 | 80.29 | 79.99 | 96.37 | 55.98 | 56.47 | 44.74 |
| A4 | +DG-FPN+OLDM | 71.14 | **81.44** | 81.75 | 96.73 | 56.15 | 58.79 | 48.13 |
| A5 | +SGFR+SG-PPM | 70.90 | 81.23 | 81.22 | 96.52 | 55.56 | 58.08 | 48.02 |
| A6 | +SGFR+DG-FPN+OLDM | 71.20 | 81.49 | 83.19 | 96.49 | **56.53** | 58.42 | 48.38 |
| A7 | +SG-PPM+DG-FPN+OLDM | 71.05 | 81.38 | 82.28 | 96.46 | 55.78 | 58.32 | 48.64 |
| **Full** | **+SGFR+SG-PPM+DG-FPN+OLDM** | **71.40** | **81.63** | 80.85 | **96.90** | 56.07 | **59.18** | 48.62 |

**With 8× multi-scale test-time augmentation (`slide_tta_ms`)**

| ID | Configuration | mIoU | mDice | mAcc | aAcc | oil_spill | look_alike | ship |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| A1 | ConvNeXtV2 | 71.49 | 81.68 | 81.30 | 96.64 | 56.49 | 59.56 | 48.07 |
| A2 | +SGFR | 70.12 | 80.52 | 81.07 | 96.30 | 55.99 | 55.81 | 45.57 |
| A3 | +SG-PPM | 70.64 | 80.95 | 81.78 | 96.47 | 56.26 | 57.85 | 45.88 |
| A4 | +DG-FPN+OLDM | 71.83 | 81.96 | 83.39 | 96.80 | 56.24 | 60.64 | 48.92 |
| A5 | +SGFR+SG-PPM | 71.47 | 81.69 | 82.75 | 96.48 | 56.21 | 58.62 | **49.28** |
| A6 | +SGFR+DG-FPN+OLDM | 71.57 | 81.79 | **84.44** | 96.41 | **57.23** | 58.33 | 49.25 |
| A7 | +SG-PPM+DG-FPN+OLDM | 71.53 | 81.75 | 83.47 | 96.47 | 56.36 | 58.76 | 49.44 |
| **Full** | **+SGFR+SG-PPM+DG-FPN+OLDM** | **71.99** | **82.09** | 82.03 | **96.98** | 56.30 | **60.54** | 49.49 |

> **Honest reading of the ablation.** SGFR and SG-PPM are *not* positive
> contributors when added on their own (A2, A3 are below A1). The gain comes
> from the **combination**: DG-FPN + OLDM is the single strongest increment
> (`+0.68` mIoU over A1, `+1.69` on `ship`), and the full model reaches the
> best mIoU / mDice / `look_alike` IoU. A2 and A3 remain available as
> independently switchable components, and `use_sgppm` / `use_oldm` can be
> toggled in the config.

### 2. Comparison against standard segmentation baselines

Trained for 150 epochs on the same dataset. Metrics are the **best validation
mIoU** recorded in each run's training log.

| Model | Best mIoU | Best epoch | mIoU @ final epoch |
|:--|--:|--:|--:|
| **UPerNet ConvNeXt V2-T** (backbone baseline) | **68.69** | 60 | 67.82 |
| SegNeXt-B (MSCAN-B) | 67.21 | 70 | 65.17 |
| UPerNet Swin-T | 66.93 | 85 | 64.49 |
| DeepLabV3+ R50 | 65.54 | 95 | 63.81 |
| SegFormer MIT-B2 | 64.19 | 75 | 60.24 |
| UNet | 61.58 | 55 | 59.87 |

Per-class IoU at each model's best epoch — the `oil_spill` class is where the
baselines actually differ:

| Class | ConvNeXtV2-T | SegNeXt-B | UPerNet Swin-T | DeepLabV3+ | SegFormer-B2 | UNet |
|:--|--:|--:|--:|--:|--:|--:|
| sea_surface | 96.82 | 96.66 | 96.73 | 96.57 | 96.74 | 94.86 |
| **oil_spill** | **52.73** | 49.59 | 48.75 | 49.07 | 47.58 | 48.76 |
| look_alike | 61.37 | 60.19 | **61.53** | 60.04 | 60.54 | 39.34 |
| ship | **36.93** | 34.84 | 35.33 | 29.49 | 22.63 | 32.95 |
| land | 95.61 | 94.76 | 92.29 | 92.55 | 93.46 | 92.00 |
| **mIoU** | **68.69** | 67.21 | 66.93 | 65.54 | 64.19 | 61.58 |

> ⚠️ **Metric provenance.** The baseline table comes from MMEngine validation
> logs of the `mmsegmentation` runs (the dataset's `test/` split is used as the
> validation set there), **not** from an independent `tools/test.py` script.
> The ablation table in §1 comes from this standalone repository's own
> evaluation pipeline. Treat the two tables as **different protocols** and do
> not read a cross-table delta as a like-for-like improvement.

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare the data

`OilSpillDataset` expects the following layout:

```
Oil-Spill-Detection/
├── train/
│   ├── images/       # .jpg   SAR images
│   ├── labels/       # .png   RGB mask (black=sea, cyan=oil, red=look-alike, brown=ship, green=land)
│   └── labels_1D/    # .png   1-D label (0=sea, 1=oil, 2=look-alike, 3=ship, 4=land)
└── test/
    ├── images/
    ├── labels/
    └── labels_1D/
```

Then point `data.root` in the YAML config at that directory and set
`model.backbone.pretrained` to a local ConvNeXt V2 checkpoint path.

### 3. Train

```bash
# single GPU
python tools/train.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml

# 2-GPU DDP
python tools/train.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml --launcher pytorch

# or use the provided tmux launcher (trains, then auto-runs slide / TTA tests)
bash tools/train_sgfr_dgfpn_oldm_b_tmux_2gpu.sh
```

`--work-dir` overrides the output directory from the config, and `--amp`
enables mixed precision.

### 4. Evaluate

```bash
# sliding-window evaluation
python tools/test.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml \
    --checkpoint work_dirs/<your_run>/best_mIoU.pth --slide --out-dir outputs/test

# plain (non-slide) evaluation
python tools/test.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml \
    --checkpoint work_dirs/<your_run>/best_mIoU.pth

# TTA / multi-scale TTA
python tools/test_tta.py --config <config> --checkpoint <ckpt>
python tools/test_tta_ms.py --config <config> --checkpoint <ckpt>
```

Add `--save-md results.md` to `tools/test.py` to append a markdown metric
table to a file.

### 5. Inference

```bash
python tools/infer.py \
    --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml \
    --checkpoint work_dirs/<your_run>/best_mIoU.pth \
    --input /path/to/image_or_dir \
    --output outputs/infer
```

### 6. Unit tests

```bash
pytest tests/
```

---

## Repository layout

```
sde-net-sar-oil-spill-segmentation/
├── configs/                    # YAML configs
│   ├── convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml   # ← SDE-Net (main)
│   ├── convnextv2_t_upernet_oilspill.yaml                   # plain UPerNet baseline
│   └── convnextv2_t_enhanced_upernet_oilspill.yaml          # enhanced variant
├── datasets/
│   ├── oil_spill.py            # OilSpillDataset (5 classes)
│   └── transforms.py           # TrainTransform / EvalTransform
├── engine/
│   ├── trainer.py  evaluator.py  optimizer.py
│   ├── inference.py            # plain inference
│   ├── inference_tta.py        # TTA inference
│   └── inference_tta_ms.py     # multi-scale TTA inference
├── models/
│   ├── backbone/convnext_v2.py
│   ├── decoder/                # UPerHead + SGFR / DG-FPN / OLDM / OLDM variants
│   ├── modules/                # SGFR, SGPPM, OLDM, DySample, LGDA, SPCM,
│   │                           # ClassAwareGatedFPN, PSAB, SMFF, GDAFM, ...
│   ├── losses.py               # CE + Dice, with class weights
│   └── segmentor*.py           # one segmentor per decode-head type
├── tests/                      # pytest unit tests for the custom modules
├── tools/                      # train.py / test.py / infer.py + 2-GPU tmux launchers
├── utils/                      # config, checkpoint, distributed, visualize
├── docs/README_original.md     # original project README (kept for reference)
├── ENHANCED_MODEL.md           # enhanced model: LGDA + SPCM + ClassAwareGatedFPN
├── CLAUDE.md                   # repo guide for coding agents
└── requirements.txt
```

### Switching models

`tools/train.py` dispatches on `model.decode_head.type`:

| `decode_head.type` | Segmentor |
|:--|:--|
| `UPerHead` | `models/segmentor.py` → `EncoderDecoder` |
| `UPerHeadSGFRDGFPNOLDM` | `models/segmentor_sgfr_dgfpn_oldm.py` → **SDE-Net** |
| `UPerHeadDGFPNOLDM` | `models/segmentor_dg_fpn_oldm.py` |
| `UPerHeadOLDM` | `models/segmentor_oldm.py` |
| `UPerHeadSGFR` | `models/segmentor_sgfr.py` |
| `UPerHeadOLDMGatedTopDown` | `models/segmentor_oldm_gated_td.py` |
| `UPerHeadOLDMPartialCH` | `models/segmentor_oldm_partial_ch.py` |
| `UPerHeadOLDMLgdaSpcm` | `models/segmentor_oldm_lgda_spcm.py` |
| `UPerHeadOLDMSpcm` | `models/segmentor_oldm_spcm.py` |
| `UPerHeadPSABSMFFOLDM` | `models/segmentor_psab_smff_oldm.py` |
| `LiteGFPNOLDMHead` | `models/segmentor_lite_gfpn_oldm.py` |
| `UPerHeadAblation` | `models/segmentor_ablation.py` |
| `EnhancedUPerHead` | `models/segmentor_enhanced.py` (see [`ENHANCED_MODEL.md`](ENHANCED_MODEL.md)) |

### Classes and loss configuration

```yaml
class_names: [sea_surface, oil_spill, look_alike, ship, land]

model:
  loss:
    ce_weight: 1.0
    dice_weight: 2.0
    class_weights: [0.5, 2.6, 1.5, 3.0, 0.5]   # sea, oil, look-alike, ship, land
    ignore_index: 255
```

`oil_spill` and `ship` are up-weighted because they are the small / hard
classes; `sea_surface` and `land` are down-weighted so the dominant background
cannot monopolise the optimisation.

### Key configuration switches (SDE-Net)

| Key | Default | Meaning |
|:--|:--|:--|
| `model.decode_head.use_sgppm` | `true` | SG-PPM instead of vanilla PPM on `C4` |
| `model.decode_head.use_oldm` | `true` | Enable the OLDM disambiguation branch |
| `model.decode_head.sgfr_stripe_k` | `7` | Strip kernel size inside SGFR |
| `model.decode_head.sgppm_reduction` | `16` | Channel reduction of the SG-PPM gate |
| `model.decode_head.oldm_reduction` | `16` | Channel reduction inside OLDM |
| `model.aux_weight` | `0.3` | Weight of the auxiliary FCN head |
| `test.overlap` | `0.2` | Sliding-window overlap |

---

## 中文说明

**SDE-Net** 是面向 **SAR 海上溢油语义分割** 的改进型编码器—解码器网络。
主干网络为 **ConvNeXt V2**，解码头在 UPerNet 基础上针对 SAR 溢油分割的三个
核心难点做了重构：

| 模块 | 位置 | 解决的问题 |
|:--|:--|:--|
| **SGFR** 条带引导特征细化 | 浅层侧支 `C1/C2/C3` | 浅层相干斑噪声与模糊边界 |
| **SG-PPM** 尺度门控金字塔池化 | 深层特征 `C4` | 不同上下文尺度的重要性自适应加权 |
| **DG-FPN** DySample 门控特征金字塔 | 自顶向下融合路径 | 双线性上采样的细节丢失、朴素相加的融合缺陷 |
| **OLDM** 油–相似像判别模块 | 融合后、分类前 | 溢油与「类油」暗斑在低后向散射下几乎同签名 |

**分类（5 类）**：`sea_surface`（海面）、`oil_spill`（溢油）、`look_alike`
（相似像）、`ship`（船舶）、`land`（陆地）。损失函数为 **CE + Dice**，并对
`oil_spill`、`ship` 等小类做类别加权。

**快速上手**

```bash
pip install -r requirements.txt
# 修改 config 里的 data.root 与 backbone.pretrained
python tools/train.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml --launcher pytorch
python tools/test.py  --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml \
                      --checkpoint work_dirs/<你的实验>/best_mIoU.pth --slide
python tools/infer.py --config configs/convnextv2_b_sgfr_dgfpn_oldm_upernet_oilspill.yaml \
                      --checkpoint work_dirs/<你的实验>/best_mIoU.pth \
                      --input /path/to/images --output outputs/infer
```

**结果说明**：消融实验表来自本仓库自带的评测流程；基线对比表来自
`mmsegmentation` 训练日志中的验证指标，两者评测口径不同，请勿直接相减。

---

## Acknowledgements

- **ConvNeXt V2** — *A ConvNet for the 2020s* (Liu et al., CVPR 2022)
- **UPerNet** — *Unified Perceptual Parsing for Scene Understanding* (Xiao et al., ECCV 2018)
- **DySample** — *Learning to Upsample by Learning to Sample* (Huang et al., ICCV 2023)
- **Oil-Spill-Detection dataset** — SAR oil spill segmentation benchmark
  (1002 training / 110 testing samples, 5 classes)

---

## License

Released for academic and research use. Please cite the corresponding paper
when you use this code or the trained weights.
