#!/usr/bin/env bash
set -euo pipefail

ROOT="/run/media/sdd1/OilSeg/ConvNeXtV2-UPerNet-OilSpill"
SESSION="${TMUX_SESSION:-convnextv2_t_upernet_oilspill_2gpu}"
WORKDIR="${WORKDIR:-work_dirs/convnextv2_t_upernet_oilspill}"
CONFIG="${CONFIG:-configs/convnextv2_t_upernet_oilspill.yaml}"
PYTHON="${PYTHON:-/run/media/sdd1/OilSeg/.venv/bin/python}"
PORT="${PORT:-29623}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
mkdir -p "$WORKDIR"

# region agent log
"$PYTHON" - <<PY
import json, time
payload = {
    "sessionId": "f8a0ed",
    "runId": "post-fix-2",
    "hypothesisId": "H1",
    "location": "tools/train_tmux_2gpu.sh:startup",
    "message": "tmux launcher parameters",
    "data": {
        "root": "$ROOT",
        "session": "$SESSION",
        "workdir": "$WORKDIR",
        "config": "$CONFIG",
        "python": "$PYTHON",
        "port": "$PORT",
        "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
    },
    "timestamp": int(time.time() * 1000),
}
with open("/run/media/sdd1/OilSeg/.cursor/debug-f8a0ed.log", "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
# endregion

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

CMD="cd '$ROOT' && mkdir -p '$WORKDIR' && exec > >(tee -a '$WORKDIR/train_tmux.log') 2>&1 && \
echo '[agent] ROOT=$ROOT' && \
echo '[agent] PYTHON=$PYTHON' && \
echo '[agent] CONFIG=$CONFIG' && \
echo '[agent] WORKDIR=$WORKDIR' && \
echo '[agent] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES' && \
echo '[agent] PORT=$PORT' && \
'$PYTHON' -c 'import sys, torch; print(\"[agent] sys.executable=\", sys.executable); print(\"[agent] torch=\", torch.__version__)' && \
'$PYTHON' -c \"import json,time; payload={'sessionId':'f8a0ed','runId':'post-fix-3','hypothesisId':'H1','location':'tools/train_tmux_2gpu.sh:tmux-command','message':'tmux command reached torch preflight','data':{'python':'$PYTHON','port':'$PORT'},'timestamp':int(time.time()*1000)}; open('/run/media/sdd1/OilSeg/.cursor/debug-f8a0ed.log','a',encoding='utf-8').write(json.dumps(payload,ensure_ascii=False)+'\n')\" && \
'$PYTHON' -m torch.distributed.run --nproc_per_node=2 --master_port='$PORT' tools/train.py \
  --config '$CONFIG' --work-dir '$WORKDIR' --launcher pytorch --amp"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
