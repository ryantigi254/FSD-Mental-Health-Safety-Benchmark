## LM Studio helper scripts

- `copy_lmstudio_log.ps1`: After running generations, snapshot the current LM Studio `main.log` (from `%APPDATA%\LM Studio\logs\`) into `results/gpt-oss-20b/lmstudio-main-<timestamp>.log` for later correlation with JSONL outputs.

Notes:
- No elevation required; the script copies from your roaming profile into the repo `results` tree.
- Ensure LM Studio has been running so `main.log` exists before invoking.

