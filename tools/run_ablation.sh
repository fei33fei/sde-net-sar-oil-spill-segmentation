#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/run/media/sdd1/OilSeg/.venv/bin/python}"
ABLATION_DIR="work_dirs/ablation"
SUMMARY_MD="${ABLATION_DIR}/消融实验结果.md"
NGPU="${NGPU:-1}"
PORT="${PORT:-29700}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$ABLATION_DIR"

CONFIGS=(
    "configs/ablation/A1_convnextv2b_upernet.yaml"
    "configs/ablation/A2_convnextv2b_sgfr.yaml"
    "configs/ablation/A3_convnextv2b_sgppm.yaml"
    "configs/ablation/A4_convnextv2b_dgfpn_oldm.yaml"
    "configs/ablation/A5_convnextv2b_sgfr_sgppm.yaml"
    "configs/ablation/A6_convnextv2b_sgfr_dgfpn_oldm.yaml"
    "configs/ablation/A7_convnextv2b_sgppm_dgfpn_oldm.yaml"
)

NAMES=(
    "A1_convnextv2b_upernet"
    "A2_convnextv2b_sgfr"
    "A3_convnextv2b_sgppm"
    "A4_convnextv2b_dgfpn_oldm"
    "A5_convnextv2b_sgfr_sgppm"
    "A6_convnextv2b_sgfr_dgfpn_oldm"
    "A7_convnextv2b_sgppm_dgfpn_oldm"
)

LABELS=(
    "A1: base+1 (ConvNeXtV2)"
    "A2: base+1+2 (ConvNeXtV2+SGFR)"
    "A3: base+1+3 (ConvNeXtV2+SG-PPM)"
    "A4: base+1+4 (ConvNeXtV2+DG-FPN+OLDM)"
    "A5: base+1+2+3 (ConvNeXtV2+SGFR+SG-PPM)"
    "A6: base+1+2+4 (ConvNeXtV2+SGFR+DG-FPN+OLDM)"
    "A7: base+1+3+4 (ConvNeXtV2+SG-PPM+DG-FPN+OLDM)"
)

cat > "$SUMMARY_MD" <<'HEADER'
# 消融实验结果汇总

| 编号 | 改进点 | backbone | SGFR | SG-PPM | DG-FPN+OLDM |
|------|--------|----------|------|--------|-------------|
| A0 | base | Res50 | - | - | - |
| A1 | +1 | ConvNeXtV2 | - | - | - |
| A2 | +1+2 | ConvNeXtV2 | ✓ | - | - |
| A3 | +1+3 | ConvNeXtV2 | - | ✓ | - |
| A4 | +1+4 | ConvNeXtV2 | - | - | ✓ |
| A5 | +1+2+3 | ConvNeXtV2 | ✓ | ✓ | - |
| A6 | +1+2+4 | ConvNeXtV2 | ✓ | - | ✓ |
| A7 | +1+3+4 | ConvNeXtV2 | - | ✓ | ✓ |

HEADER

train_one() {
    local config="$1"
    echo "========================================"
    echo "[train] $config  (NGPU=$NGPU)"
    echo "========================================"
    if [ "$NGPU" -gt 1 ]; then
        "$PYTHON" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$PORT" \
            tools/train.py --config "$config" --launcher pytorch --amp
    else
        "$PYTHON" tools/train.py --config "$config" --amp
    fi
}

test_one() {
    local config="$1"
    local work_dir="$2"
    local name="$3"
    local label="$4"
    local ckpt="${work_dir}/best_mIoU.pth"
    local md_file="${work_dir}/测试结果.md"

    if [ ! -f "$ckpt" ]; then
        echo "[WARN] checkpoint not found: $ckpt — skipping tests for $name"
        return
    fi

    echo "# ${label}" > "$md_file"

    echo "[test] slide (mean_logits + vote) — $name"
    "$PYTHON" tools/test.py \
        --config "$config" --checkpoint "$ckpt" \
        --slide --slide-aggregate both --save-md "$md_file"

    echo "[test] slide_tta — $name"
    "$PYTHON" tools/test_tta.py \
        --config "$config" --checkpoint "$ckpt" \
        --slide --save-md "$md_file"

    echo "[test] slide_tta_ms — $name"
    "$PYTHON" tools/test_tta_ms.py \
        --config "$config" --checkpoint "$ckpt" \
        --slide --save-md "$md_file"
}

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    label="${LABELS[$i]}"
    work_dir="${ABLATION_DIR}/${name}"

    train_one "$config"
    test_one "$config" "$work_dir" "$name" "$label"
done

echo ""
echo "========================================"
echo "  汇总各实验 best mIoU"
echo "========================================"

{
    echo ""
    echo "## slide/mean_logits mIoU 汇总"
    echo ""
    echo "| 编号 | 配置 | mIoU | oil_spill IoU | look_alike IoU | ship IoU |"
    echo "|------|------|------|---------------|----------------|----------|"
} >> "$SUMMARY_MD"

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    label="${LABELS[$i]}"
    md_file="${ABLATION_DIR}/${name}/测试结果.md"
    if [ -f "$md_file" ]; then
        miou=$(grep -A3 '### slide/mean_logits' "$md_file" | grep 'mIoU' | head -1 | sed 's/.*\*\*\([0-9.]*\)\*\*.*/\1/')
        oil=$(grep -A12 '### slide/mean_logits' "$md_file" | grep 'oil_spill' | head -1 | awk -F'|' '{print $3}' | xargs)
        look=$(grep -A12 '### slide/mean_logits' "$md_file" | grep 'look_alike' | head -1 | awk -F'|' '{print $3}' | xargs)
        ship=$(grep -A12 '### slide/mean_logits' "$md_file" | grep 'ship' | head -1 | awk -F'|' '{print $3}' | xargs)
        echo "| ${name%%_*} | ${label} | ${miou:-N/A} | ${oil:-N/A} | ${look:-N/A} | ${ship:-N/A} |" >> "$SUMMARY_MD"
    else
        echo "| ${name%%_*} | ${label} | N/A | N/A | N/A | N/A |" >> "$SUMMARY_MD"
    fi
done

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    label="${LABELS[$i]}"
    md_file="${ABLATION_DIR}/${name}/测试结果.md"
    if [ -f "$md_file" ]; then
        {
            echo ""
            echo "---"
            echo ""
            echo "## ${label}"
            echo ""
            cat "$md_file"
        } >> "$SUMMARY_MD"
    fi
done

echo ""
echo "[done] 汇总文件: $SUMMARY_MD"
