#!/usr/bin/env python3
"""
Cross-platform generation launcher for Study A/B/C/Bias.

Purpose:
- Avoid OS-specific absolute paths in docs/commands.
- Run the correct generation script from any working directory.
- Optionally run through `conda run -n <env>` without manual activation.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run study generation scripts with automatic Uni-setup root resolution."
    )
    parser.add_argument(
        "--study",
        required=True,
        choices=["study_a", "study_a_bias", "study_b", "study_b_multi_turn", "study_c"],
        help="Study generation target.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID for the target study script.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Optional conda env name. If omitted, uses current Python interpreter.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate path/script/model wiring only. Does not execute generation.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> int:
    args, passthrough = parse_args()

    uni_setup_root = Path(__file__).resolve().parents[2]

    study_script_map = {
        "study_a": uni_setup_root / "hf-local-scripts" / "run_study_a_generate_only.py",
        "study_a_bias": uni_setup_root / "hf-local-scripts" / "run_study_a_bias_generate_only.py",
        "study_b": uni_setup_root / "hf-local-scripts" / "run_study_b_generate_only.py",
        "study_b_multi_turn": uni_setup_root / "hf-local-scripts" / "run_study_b_multi_turn_generate_only.py",
        "study_c": uni_setup_root / "hf-local-scripts" / "run_study_c_generate_only.py",
    }
    allowed_model_ids_by_study = {
        "study_a": {
            "qwen3_lmstudio",
            "qwq",
            "deepseek_r1_lmstudio",
            "gpt_oss",
            "psyllm_gml_local",
            "piaget_local",
            "psyche_r1_local",
            "psych_qwen_local",
        },
        "study_a_bias": {
            "qwen3_lmstudio",
            "qwq",
            "deepseek_r1_lmstudio",
            "gpt_oss_lmstudio",
            "psyllm_gml_local",
            "piaget_local",
            "psyche_r1_local",
            "psych_qwen_local",
            "psyllm",
        },
        "study_b": {
            "qwen3_lmstudio",
            "qwq",
            "deepseek_r1_lmstudio",
            "gpt_oss",
            "psyllm_gml_local",
            "piaget_local",
            "psyche_r1_local",
            "psych_qwen_local",
            "psyllm",
        },
        "study_b_multi_turn": {
            "qwen3_lmstudio",
            "qwq",
            "deepseek_r1_lmstudio",
            "gpt_oss",
            "psyllm_gml_local",
            "piaget_local",
            "psyche_r1_local",
            "psych_qwen_local",
            "psyllm",
        },
        "study_c": {
            "qwen3_lmstudio",
            "qwq",
            "deepseek_r1_lmstudio",
            "gpt_oss",
            "psyllm_gml_local",
            "piaget_local",
            "psyche_r1_local",
            "psych_qwen_local",
            "psyllm",
        },
    }

    target_script = study_script_map[args.study]
    if not target_script.exists():
        print(f"Script not found: {target_script}", file=sys.stderr)
        return 2
    if args.model_id not in allowed_model_ids_by_study[args.study]:
        allowed_values = ", ".join(sorted(allowed_model_ids_by_study[args.study]))
        print(
            f"Model ID '{args.model_id}' is not allowed for {args.study}. "
            f"Allowed: {allowed_values}",
            file=sys.stderr,
        )
        return 2

    if args.env:
        command = ["conda", "run", "-n", args.env, "python"]
    else:
        command = [sys.executable]

    command.extend([str(target_script), "--model-id", args.model_id])
    command.extend(passthrough)

    print(f"Uni-setup root: {uni_setup_root}")
    print(f"Executing: {shlex.join(command)}")
    if args.check_only:
        print("Check-only mode: validation passed; no generation executed.")
        return 0

    completed = subprocess.run(command, cwd=str(uni_setup_root))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
