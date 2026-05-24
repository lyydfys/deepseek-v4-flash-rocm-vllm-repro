#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=${BASE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
MODEL=${MODEL:-deepseek-v4-flash-amd-32k-batch8-16384}
TOPKS=${TOPKS:-1024 1536 2048 4096}
LOG_ROOT=${LOG_ROOT:-/home/amd_research_logs/8k_topk_sweep_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$LOG_ROOT"

wait_models() {
  local deadline=$((SECONDS + 1800))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -sS --max-time 8 http://127.0.0.1:8000/v1/models >/tmp/models_check.json 2>/tmp/models_check.err; then
      cat /tmp/models_check.json > "$LOG_ROOT/models_latest.json"
      return 0
    fi
    sleep 10
  done
  return 1
}

echo "SWEEP_START time=$(date '+%F %T %Z') topks=$TOPKS log_root=$LOG_ROOT"

for topk in $TOPKS; do
  echo "TOPK_START topk=$topk time=$(date '+%F %T %Z')"
  bash "$BASE/scripts/patch_home_model_config_index_topk.sh" "$topk" 2>&1 | tee "$LOG_ROOT/patch_topk_${topk}.log"

  pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  sleep 3
  launch_log="$LOG_ROOT/launch_topk_${topk}.log"
  nohup bash "$BASE/scripts/launch_32k_batch8_16384_system.sh" > "$launch_log" 2>&1 &
  echo "TOPK_LAUNCH topk=$topk pid=$! log=$launch_log"

  if ! wait_models; then
    echo "TOPK_READY_FAIL topk=$topk"
    tail -n 120 "$launch_log" || true
    continue
  fi

  gate_log="$LOG_ROOT/gate_8k_topk_${topk}.log"
  set +e
  python3 "$BASE/scripts/run_openai_gate.py" \
    --model "$MODEL" \
    --name needle8192_tokens \
    --timeout 1200 \
    2>&1 | tee "$gate_log"
  gate_rc=${PIPESTATUS[0]}

  stream_log="$LOG_ROOT/stream_8k_topk_${topk}.log"
  stream_jsonl="$LOG_ROOT/stream_8k_topk_${topk}.jsonl"
  python3 "$BASE/scripts/run_stream_bench.py" \
    --model "$MODEL" \
    --target-tokens 8192 \
    --max-tokens 1 \
    --runs 1 \
    --timeout 1200 \
    --output-jsonl "$stream_jsonl" \
    2>&1 | tee "$stream_log"
  stream_rc=${PIPESTATUS[0]}
  set -e

  echo "TOPK_RESULT topk=$topk gate_rc=$gate_rc stream_rc=$stream_rc"
done

echo "SWEEP_DONE time=$(date '+%F %T %Z') log_root=$LOG_ROOT"
