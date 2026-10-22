#!/usr/bin/env bash
set -euo pipefail

# ConvNeXtV2-Base: 150ep, lr=6e-5, warmup=10, crop=512, whole_resize_prob=0.3
# + auto slide test & TTA test
ROOT="/run/media/sdd1/OilSeg/ConvNeXtV2-UPerNet-OilSpill"
SESSION="${TMUX_SESSION:-enhanced_base_2gpu}"
WORKDIR="${WORKDIR:-work_dirs/convnextv2_b_enhanced_upernet_oilspill}"
CONFIG="${CONFIG:-configs/convnextv2_b_enhanced_upernet_oilspill.yaml}"
PYTHON="${PYTHON:-/run/media/sdd1/OilSeg/.venv/bin/python}"
PORT="${PORT:-29628}"
MD_FILE="${WORKDIR}/测试结果.md"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
mkdir -p "$WORKDIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

CMD="cd '$ROOT' && mkdir -p '$WORKDIR' && exec > >(tee -a '$WORKDIR/train_tmux.log') 2>&1 && \
echo '[base] ROOT=$ROOT' && \
echo '[base] PYTHON=$PYTHON' && \
echo '[base] CONFIG=$CONFIG' && \
echo '[base] WORKDIR=$WORKDIR' && \
echo '[base] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES' && \
echo '[base] PORT=$PORT' && \
'$PYTHON' -c 'import sys, torch; print(\"[base] sys.executable=\", sys.executable); print(\"[base] torch=\", torch.__version__)' && \
'$PYTHON' -m torch.distributed.run --nproc_per_node=2 --master_port='$PORT' tools/train.py \
  --config '$CONFIG' --work-dir '$WORKDIR' --launcher pytorch --amp && \
echo '========== Training finished, starting evaluation ==========' && \
echo '# 测试结果' > '$MD_FILE' && \
echo '' >> '$MD_FILE' && \
echo '## 普通 Slide 测试 (best_mIoU)' >> '$MD_FILE' && \
'$PYTHON' tools/test.py \
  --config '$CONFIG' \
  --checkpoint '$WORKDIR/best_mIoU.pth' \
  --out-dir '$WORKDIR/test_best' \
  --slide --slide-aggregate both \
  --save-md '$MD_FILE' && \
echo '========== Slide test done, starting TTA test ==========' && \
echo '' >> '$MD_FILE' && \
echo '## TTA Slide 测试 (best_mIoU)' >> '$MD_FILE' && \
'$PYTHON' tools/test_tta.py \
  --config '$CONFIG' \
  --checkpoint '$WORKDIR/best_mIoU.pth' \
  --out-dir '$WORKDIR/test_tta' \
  --slide \
  --save-md '$MD_FILE' && \
echo '========== All tests done. Results saved to $MD_FILE =========='
"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "Started tmux session: $SESSION"
echo "WORKDIR: $WORKDIR"
echo "CONFIG: $CONFIG"
echo "Attach: tmux attach -t $SESSION"
