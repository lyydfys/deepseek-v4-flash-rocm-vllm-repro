#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=${BASE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
MODEL_DIR=${MODEL_DIR:-/home/deepseek_hybrid/DeepSeek-V4-Flash}
CONFIG=${CONFIG:-$MODEL_DIR/config.json}
INDEX_TOPK=${1:-${INDEX_TOPK:-2048}}
BACKUP_DIR="$BASE/patch_backups/model_config_index_topk_$(date +%Y%m%d_%H%M%S)"

if [ ! -f "$CONFIG" ]; then
  echo "CONFIG_NOT_FOUND=$CONFIG" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
cp -f "$CONFIG" "$BACKUP_DIR/config.json.bak"

export CONFIG INDEX_TOPK
python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CONFIG"])
value = int(os.environ["INDEX_TOPK"])
data = json.loads(path.read_text())
old = data.get("index_topk")
data["index_topk"] = value
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(f"INDEX_TOPK_PATCHED old={old} new={value} file={path}")
PY

python3 - <<'PY'
import json
import os
data = json.load(open(os.environ["CONFIG"]))
print("VERIFY index_topk", data.get("index_topk"))
PY
