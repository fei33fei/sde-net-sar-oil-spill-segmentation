#!/usr/bin/env bash
set -euo pipefail

# v3: strip_kernel_size=7, epochs=80, grad_clip=1.0, EMA, vertical flip + random rotation
ROOT="/run/media/sdd1/OilSeg/ConvNeXtV2-UPerNet-OilSpill"
SESSION="${TMUX_SESSION:-enhanced_v3_2gpu}"
WORKDIR="${WORKDIR:-work_dirs/convnextv2_t_enhanced_upernet_oilspill_v3}"
CONFIG="${CONFIG:-configs/convnextv2_t_enhanced_upernet_oilspill.yaml}"
PYTHON="${PYTHON:-/run/media/sdd1/OilSeg/.venv/bin/python}"
PORT="${PORT:-29625}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
mkdir -p "$WORKDIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

CMD="cd '$ROOT' && mkdir -p '$WORKDIR' && exec > >(tee -a '$WORKDIR/train_tmux.log') 2>&1 && \
echo '[v3] ROOT=$ROOT' && \
echo '[v3] PYTHON=$PYTHON' && \
echo '[v3] CONFIG=$CONFIG' && \
echo '[v3] WORKDIR=$WORKDIR' && \
echo '[v3] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES' && \
echo '[v3] PORT=$PORT' && \
'$PYTHON' -c 'import sys, torch; print(\"[v3] sys.executable=\", sys.executable); print(\"[v3] torch=\", torch.__version__)' && \
'$PYTHON' -m torch.distributed.run --nproc_per_node=2 --master_port='$PORT' tools/train.py \
  --config '$CONFIG' --work-dir '$WORKDIR' --launcher pytorch --amp"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "Started tmux session: $SESSION"
echo "WORKDIR: $WORKDIR"
echo "Attach: tmux attach -t $SESSION"
