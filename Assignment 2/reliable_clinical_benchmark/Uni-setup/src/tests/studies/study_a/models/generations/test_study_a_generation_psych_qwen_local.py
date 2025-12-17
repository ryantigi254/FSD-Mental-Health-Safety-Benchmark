"""
Smoke test for local Psych_Qwen_32B (transformers, no LM Studio).

NOTE: Do not run this while a GPU-heavy Study A job is running.

Run (PowerShell):
    cd Uni-setup
    $Env:PYTHONNOUSERSITE="1"
    $Env:PYTHONPATH="src"
    $py="C:\\Users\\22837352\\.conda\\envs\\mh-llm-benchmark-env\\python.exe"
    & $py src\\tests\\test_psych_qwen_local.py --prompt-idx 1
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from typing import Dict, Any

from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner
from reliable_clinical_benchmark.models.base import GenerationConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Compumacy/Psych_Qwen_32B",
        help="HF model id or local directory (e.g. models/Psych_Qwen_32B).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        help="Quantization mode: 4bit (NF4), 8bit, or none.",
    )
    parser.add_argument(
        "--prompt-idx",
        type=int,
        default=0,
        help="1-based index of a single prompt to run (0 runs all prompts).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="If >0, only run the first N prompts.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max new tokens per prompt.",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip printing/storing raw output text (faster).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    runner = PsychQwen32BLocalRunner(
        model_name=args.model,
        quantization=args.quantization,
        config=GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            max_tokens=args.max_tokens,
        ),
    )

    prompts = [
        "Give me a one-line clinical differential for persistent low mood.",
        "List two bullet coping tips for test anxiety.",
        "Explain why empty tool calls might appear; keep it short.",
        "Provide a brief step-by-step reasoning (<think> tags allowed) for insomnia causes.",
        "State one concise diagnosis guess for: 'throat tightness and shaking when anxious'.",
    ]

    if args.prompt_idx:
        if args.prompt_idx < 1 or args.prompt_idx > len(prompts):
            raise SystemExit(f"--prompt-idx must be 1..{len(prompts)}")
        prompts = [prompts[args.prompt_idx - 1]]
    elif args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    for i, p in enumerate(prompts, 1):
        print(f"--- Prompt {i} (starting) ---", flush=True)
        print(p, flush=True)
        sys.stdout.flush()
        t0 = time.perf_counter()
        try:
            out_cot = runner.generate(p, mode="cot")
            out_direct = runner.generate(p, mode="direct")
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            think_pat = r"<think>(.*?)</think>"
            think_cot = re.search(think_pat, out_cot or "", re.DOTALL)
            think_direct = re.search(think_pat, out_direct or "", re.DOTALL)
            log: Dict[str, Any] = {
                "run_id": run_id,
                "prompt_idx": i,
                "prompt": p,
                "model": runner.model_name,
                "elapsed_ms": elapsed_ms,
                "has_think_cot": bool(think_cot),
                "has_think_direct": bool(think_direct),
            }
            if not args.skip_raw:
                log["output_cot"] = (out_cot or "")[:4000]
                log["output_direct"] = (out_direct or "")[:4000]
        except Exception as e:
            log = {
                "run_id": run_id,
                "prompt_idx": i,
                "prompt": p,
                "model": runner.model_name,
                "error": str(e),
            }

        print(f"--- Prompt {i} (done) ---", flush=True)
        print(json.dumps(log, indent=2), flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()


