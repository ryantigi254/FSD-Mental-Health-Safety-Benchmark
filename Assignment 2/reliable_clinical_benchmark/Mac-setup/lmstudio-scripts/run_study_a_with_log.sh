#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_id> [extra args for run_evaluation.py]"
  exit 1
fi

MODEL="$1"; shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_SRC="$HOME/Library/Application Support/LM Studio/logs/main.log"
RESULTS_DIR="$ROOT/results/$MODEL"
LOG_DIR="$RESULTS_DIR/logs"
CACHE_PATH="$RESULTS_DIR/study_a_generations.jsonl"

mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG_DEST="$LOG_DIR/lmstudio-main-${MODEL}-${TS}.log"
CACHE_SNAPSHOT="$LOG_DIR/study_a_generations-${TS}.jsonl"

if [[ ! -f "$LOG_SRC" ]]; then
  echo "LM Studio log not found at: $LOG_SRC"
  echo "Start LM Studio and run once so main.log exists."
  exit 1
fi

(
  cd "$ROOT"
  PYTHONPATH=src python scripts/run_evaluation.py --model "$MODEL" --study A --generate-only "$@"
)

if [[ -f "$CACHE_PATH" ]]; then
  cp "$CACHE_PATH" "$CACHE_SNAPSHOT"
  echo "Snapshotted cache to $CACHE_SNAPSHOT"
else
  echo "Warning: cache not found at $CACHE_PATH (nothing copied)"
fi

cp "$LOG_SRC" "$LOG_DEST"
echo "Copied LM Studio log to $LOG_DEST"

echo "Done. Pairing: $CACHE_SNAPSHOT <-> $LOG_DEST"

