#!/usr/bin/env bash
set -euo pipefail

LOG_SRC="$HOME/Library/Application Support/LM Studio/logs/main.log"

if [[ ! -f "$LOG_SRC" ]]; then
  echo "LM Studio log not found at: $LOG_SRC"
  exit 1
fi

MODEL="${1:-psyllm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT/results/$MODEL/logs"
mkdir -p "$DEST_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
DEST="$DEST_DIR/lmstudio-main-${MODEL}-${TS}.log"

cp "$LOG_SRC" "$DEST"
echo "Copied LM Studio log to $DEST"

