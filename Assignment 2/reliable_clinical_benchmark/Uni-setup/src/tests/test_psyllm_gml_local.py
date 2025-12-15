"""
Smoke test for local HF PsyLLM (GMLHUHE/PsyLLM).

Run (PowerShell):
    cd Uni-setup
    $Env:PYTHONPATH="src"
    python src/tests/test_psyllm_gml_local.py --max-prompts 2 --max-tokens 256
"""

import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict

from reliable_clinical_benchmark.models.psyllm_gml_local import PsyLLMGMLLocalRunner
from reliable_clinical_benchmark.models.base import GenerationConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="GMLHUHE/PsyLLM", help="HF model id or local folder")
    p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per prompt")
    p.add_argument("--prompt-idx", type=int, default=0, help="1-based index; 0 runs all")
    p.add_argument("--max-prompts", type=int, default=0, help="If >0, run first N prompts")
    p.add_argument("--skip-raw", action="store_true", help="Skip printing/storing raw output")
    return p.parse_args()


def main() -> None:
    # Keep smoke output clean on Windows/CUDA builds that lack FlashAttention.
    # This warning is emitted by Transformers SDPA integration and is not a failure.
    warnings.filterwarnings(
        "ignore",
        message=r".*Torch was not compiled with flash attention.*",
        category=UserWarning,
    )

    args = _parse_args()
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    runner = PsyLLMGMLLocalRunner(
        model_name=args.model,
        config=GenerationConfig(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens),
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

    for i, ptxt in enumerate(prompts, 1):
        print(f"--- Prompt {i} (starting) ---", flush=True)
        print(ptxt, flush=True)
        sys.stdout.flush()
        t0 = time.perf_counter()

        try:
            out_cot = runner.generate(ptxt, mode="cot")
            out_direct = runner.generate(ptxt, mode="direct")
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            think_pat = r"<think>(.*?)</think>"
            has_think_cot = bool(re.search(think_pat, out_cot or "", re.DOTALL))
            has_think_direct = bool(re.search(think_pat, out_direct or "", re.DOTALL))

            log: Dict[str, Any] = {
                "run_id": run_id,
                "prompt_idx": i,
                "prompt": ptxt,
                "model": runner.model_name,
                "elapsed_ms": elapsed_ms,
                "has_think_cot": has_think_cot,
                "has_think_direct": has_think_direct,
            }
            if not args.skip_raw:
                log["output_cot"] = (out_cot or "")[:4000]
                log["output_direct"] = (out_direct or "")[:4000]
        except Exception as e:
            log = {
                "run_id": run_id,
                "prompt_idx": i,
                "prompt": ptxt,
                "model": runner.model_name,
                "error": str(e),
            }

        print(f"--- Prompt {i} (done) ---", flush=True)
        print(json.dumps(log, indent=2), flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()


