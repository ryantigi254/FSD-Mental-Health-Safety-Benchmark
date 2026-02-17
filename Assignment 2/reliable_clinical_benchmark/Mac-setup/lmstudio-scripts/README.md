## LM Studio helper scripts (Mac)

- `copy_lmstudio_log.sh <model_id>`  
  Snapshot the current LM Studio `main.log` (from `~/Library/Application Support/LM Studio/logs/`) into `results/<model_id>/logs/lmstudio-main-<model>-<timestamp>.log` for later correlation with JSONL outputs.

- `run_study_a_with_log.sh <model_id> [extra args]`  
  Run Study A generation-only and immediately snapshot both the cache and LM Studio log into `results/<model_id>/logs/` with matching timestamps (`study_a_generations-<ts>.jsonl` and `lmstudio-main-<model>-<ts>.log`). Pass any extra flags (e.g., `--max-samples 5`).

Notes:
- No elevation required; the scripts copy from your user Library into the repo `results` tree.
- Ensure LM Studio has run at least once so `main.log` exists.

