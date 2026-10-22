#!/usr/bin/env bash
set -euo pipefail

# Light enhanced model + SPCM diagonal strip convs (main + anti)
ROOT="/run/media/sdd1/OilSeg/ConvNeXtV2-UPerNet-OilSpill"
SESSION="${TMUX_SESSION:-enhanced_light_diag_2gpu}"
WORKDIR="${WORKDIR:-work_dirs/convnextv2_t_enhanced_light_diag_upernet_oilspill}"
CONFIG="${CONFIG:-configs/convnextv2_t_enhanced_light_diag_upernet_oilspill.yaml}"
PYTHON="${PYTHON:-/run/media/sdd1/OilSeg/.venv/bin/python}"
PORT="${PORT:-29628}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
mkdir -p "$WORKDIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

CMD="cd '$ROOT' && mkdir -p '$WORKDIR' && exec > >(tee -a '$WORKDIR/train_tmux.log') 2>&1 && \
echo '[light-diag] ROOT=$ROOT' && \
echo '[light-diag] PYTHON=$PYTHON' && \
echo '[light-diag] CONFIG=$CONFIG' && \
echo '[light-diag] WORKDIR=$WORKDIR' && \
echo '[light-diag] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES' && \
echo '[light-diag] PORT=$PORT' && \
'$PYTHON' -c 'import sys, torch; print(\"[light-diag] sys.executable=\", sys.executable); print(\"[light-diag] torch=\", torch.__version__)' && \
'$PYTHON' -m torch.distributed.run --nproc_per_node=2 --master_port='$PORT' tools/train.py \
  --config '$CONFIG' --work-dir '$WORKDIR' --launcher pytorch --amp"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "Started tmux session: $SESSION"
echo "WORKDIR: $WORKDIR"
echo "Attach: tmux attach -t $SESSION"
