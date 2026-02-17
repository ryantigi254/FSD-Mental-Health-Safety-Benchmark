"""
Smoke test for Study C generation-only cache.

Run (PowerShell):
    cd Uni-setup
    python src/tests/test_study_c_generate_only.py --model-id qwen3_lmstudio --max-cases 1
"""

import argparse
from datetime import datetime
from pathlib import Path

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.pipelines.study_c import run_study_c


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, default="qwen3_lmstudio")
    p.add_argument("--max-cases", type=int, default=1)
    p.add_argument("--data-dir", type=str, default="data/openr1_psy_splits")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--cache-out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)

    cache_out = args.cache_out
    if cache_out is None:
        cache_out = str(Path(args.output_dir) / args.model_id / f"study_c_generations.smoke-{run_id}.jsonl")

    run_study_c(
        model=runner,
        data_dir=args.data_dir,
        max_cases=args.max_cases,
        output_dir=args.output_dir,
        model_name=args.model_id,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
        context_cleaner="scan",
    )

    print(f"[ok] wrote {cache_out}")


if __name__ == "__main__":
    main()


