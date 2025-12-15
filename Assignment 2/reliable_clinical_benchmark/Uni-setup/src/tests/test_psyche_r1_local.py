"""
Smoke test for local Psyche-R1 (transformers, no LM Studio).

Run:
    PYTHONPATH=src python src/tests/test_psyche_r1_local.py --prompt-idx 1
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from typing import Dict, Any

import torch

from reliable_clinical_benchmark.models.psyche_r1_local import (
    PsycheR1LocalRunner,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
)
from reliable_clinical_benchmark.models.base import GenerationConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--skip-raw",
        action="store_true",
        help="Skip printing/storing the raw decoded generation (faster).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens per prompt (raise to reduce truncation).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    runner = PsycheR1LocalRunner(
        model_name="models/Psyche-R1",
        config=GenerationConfig(
            temperature=0.7,
            top_p=0.9,
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
            encoded = runner._build_inputs(p, mode="cot")
            input_token_count = int(encoded["input_ids"].shape[-1])
            with torch.inference_mode():
                gen = runner.model.generate(
                    **encoded,
                    max_new_tokens=runner.config.max_tokens,
                    do_sample=True,
                    temperature=runner.config.temperature,
                    top_p=runner.config.top_p,
                    top_k=DEFAULT_TOP_K,
                    repetition_penalty=DEFAULT_REPETITION_PENALTY,
                    eos_token_id=runner.tokenizer.eos_token_id,
                    pad_token_id=runner.tokenizer.pad_token_id,
                )
            generated_only = gen[0, input_token_count:]
            raw = runner.tokenizer.decode(generated_only, skip_special_tokens=True).strip()
            out = runner._normalize_for_mode(raw, mode="cot")

            think_match = None
            if not args.skip_raw:
                think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            log: Dict[str, Any] = {
                "run_id": run_id,
                "prompt_idx": i,
                "prompt": p,
                "model": runner.model_name,
                "elapsed_ms": elapsed_ms,
                "output_normalised": out[:4000],
            }
            if not args.skip_raw:
                log["output_raw"] = raw[:4000]
                log["has_think"] = bool(think_match)
                if think_match is not None:
                    reasoning = think_match.group(1).strip()
                    answer = raw[think_match.end() :].strip()
                    log["think_excerpt"] = reasoning[:800]
                    log["answer_excerpt"] = answer[:800]
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


