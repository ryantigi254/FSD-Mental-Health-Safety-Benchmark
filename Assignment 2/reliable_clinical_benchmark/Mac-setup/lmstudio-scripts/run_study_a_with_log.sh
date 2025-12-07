#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_id> [extra args for run_evaluation.py]"
  exit 1
fi

MODEL="$1"; shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$ROOT/results/$MODEL"
LOG_DIR="$RESULTS_DIR/logs"
CACHE_PATH="$RESULTS_DIR/study_a_generations.jsonl"

mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG_DEST="$LOG_DIR/lmstudio-main-${MODEL}-${TS}.log"
CACHE_SNAPSHOT="$LOG_DIR/study_a_generations-${TS}.jsonl"

LOG_ROOT_CANDIDATES=(
  "$HOME/Library/Application Support/LM Studio/logs"
  "$HOME/Library/Application Support/com.lmstudio.app/logs"
  "$HOME/Library/Logs/LM Studio"
)

LOG_SRC=""
for LOG_ROOT in "${LOG_ROOT_CANDIDATES[@]}"; do
  if [[ -d "$LOG_ROOT" ]]; then
    LOG_CANDIDATE="$(find "$LOG_ROOT" -maxdepth 2 -type f -name "*${MODEL}*.log" -print0 | xargs -0 ls -t 2>/dev/null | head -n1)"
    if [[ -n "$LOG_CANDIDATE" && -f "$LOG_CANDIDATE" ]]; then
      LOG_SRC="$LOG_CANDIDATE"
    elif [[ -f "$LOG_ROOT/main.log" ]]; then
      LOG_SRC="$LOG_ROOT/main.log"
    fi
  fi
  [[ -n "$LOG_SRC" ]] && break
done

if [[ -z "$LOG_SRC" || ! -f "$LOG_SRC" ]]; then
  echo "LM Studio log not found (checked common macOS paths for model-specific and main.log)."
  echo "Start LM Studio once so a log exists."
  exit 1
fi

(
  cd "$ROOT"
  PYTHONPATH="$ROOT/src" python scripts/run_evaluation.py --model "$MODEL" --study A --generate-only "$@"
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
