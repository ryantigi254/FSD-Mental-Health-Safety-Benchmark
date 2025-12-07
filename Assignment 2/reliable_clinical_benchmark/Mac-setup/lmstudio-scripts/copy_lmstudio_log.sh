#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-psyllm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT/results/$MODEL/logs"
mkdir -p "$DEST_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
DEST="$DEST_DIR/lmstudio-main-${MODEL}-${TS}.log"

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
  exit 1
fi

cp "$LOG_SRC" "$DEST"
echo "Copied LM Studio log to $DEST"

