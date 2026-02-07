#!/usr/bin/env python3
"""
Main evaluation script for Mental Health LLM Safety Benchmark.

Runs Studies A, B, and C on specified models.
"""

import argparse
import sys
import warnings
from pathlib import Path

# Suppress spacy FutureWarnings about regex set unions in newer Python versions
warnings.filterwarnings("ignore", category=FutureWarning, module="spacy")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.pipelines import study_a, study_b, study_c
from reliable_clinical_benchmark.eval.runtime_checks import (
    validate_data_files,
    validate_environment,
    validate_study_b_schema,
    check_model_availability,
)
from reliable_clinical_benchmark.utils.logging_config import setup_logging

import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Mental Health LLM Safety Benchmark Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            # Remote / LM Studio runners
            "psyllm",
            "qwq",
            "deepseek_r1",
            "deepseek_r1_lmstudio",
            "gpt_oss",
            "qwen3",
            "qwen3_lmstudio",
            # Local HF runners
            "psyllm_local",
            "psyllm_gml_local",
            "piaget_local",
            "psyche_r1_local",
            "psych_qwen_local",
            # Alternate IDs supported by the factory (kept explicit for CLI discoverability)
            "piaget-8b-local",
            "psyche-r1-local",
            "psych-qwen-32b-local",
            "psyllm-gml-local",
        ],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--study",
        type=str,
        required=True,
        choices=["A", "B", "C", "all"],
        help="Study to run (A=Faithfulness, B=Sycophancy, C=Drift, all=all studies)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to evaluate for Study C (for testing)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/openr1_psy_splits",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip runtime validation checks",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens per generation (default: 16384 to avoid truncation)",
    )
    # Study A cache/generation flags
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="(Study A) Generate outputs only and write to cache, do not compute metrics",
    )
    parser.add_argument(
        "--from-cache",
        type=str,
        default=None,
        help="(Study A) Path to cached generations JSONL to compute metrics without model calls; also used to resume",
    )
    parser.add_argument(
        "--cache-out",
        type=str,
        default=None,
        help="(Study A) Path to write cached generations JSONL when using --generate-only",
    )

    # Study B cache/generation flags
    parser.add_argument(
        "--study-b-generate-only",
        action="store_true",
        help="(Study B) Generate outputs only and write to cache, do not compute metrics",
    )
    parser.add_argument(
        "--study-b-from-cache",
        type=str,
        default=None,
        help="(Study B) Path to cached generations JSONL (metrics-from-cache mode)",
    )
    parser.add_argument(
        "--study-b-cache-out",
        type=str,
        default=None,
        help="(Study B) Path to write cached generations JSONL when using --study-b-generate-only",
    )

    # Study C cache/generation flags
    parser.add_argument(
        "--study-c-generate-only",
        action="store_true",
        help="(Study C) Generate outputs only and write to cache, do not compute metrics",
    )
    parser.add_argument(
        "--study-c-from-cache",
        type=str,
        default=None,
        help="(Study C) Path to cached generations JSONL (metrics-from-cache mode)",
    )
    parser.add_argument(
        "--study-c-cache-out",
        type=str,
        default=None,
        help="(Study C) Path to write cached generations JSONL when using --study-c-generate-only",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    logger.info("=" * 80)
    logger.info("Mental Health LLM Safety Benchmark")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Study: {args.study}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Runtime validation
    if not args.skip_checks:
        logger.info("Running validation checks...")
        env_valid, env_warnings = validate_environment()
        data_valid, missing_files = validate_data_files(
            Path(args.data_dir).parent if args.data_dir else "data"
        )

        if not data_valid:
            logger.error(f"Missing required data files: {missing_files}")
            logger.error("Please ensure all data files are present before running evaluation.")
            sys.exit(1)

        study_b_valid, study_b_errors = validate_study_b_schema(
            str(Path(args.data_dir).parent if args.data_dir else "data")
        )
        if not study_b_valid:
            logger.error("Study B split validation failed:")
            for err in study_b_errors[:20]:
                logger.error(f"  - {err}")
            if len(study_b_errors) > 20:
                logger.error(f"  ... {len(study_b_errors) - 20} more")
            sys.exit(1)

        if env_warnings:
            logger.warning("Environment validation warnings:")
            for warning in env_warnings:
                logger.warning(f"  - {warning}")

        # Check model availability
        if not check_model_availability(args.model):
            logger.warning(
                f"Model {args.model} may not be available. "
                "Continuing anyway, but evaluation may fail."
            )

    # Create model runner
    config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    try:
        model = get_model_runner(args.model, config)
        logger.info(f"Initialised model runner: {args.model}")
    except Exception as e:
        logger.error(f"Failed to initialise model runner: {e}")
        sys.exit(1)

    # Run evaluation
    studies_to_run = []
    if args.study == "all":
        studies_to_run = ["A", "B", "C"]
    else:
        studies_to_run = [args.study]

    results = {}

    for study in studies_to_run:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Running Study {study}")
        logger.info("=" * 80)

        try:
            if study == "A":
                result = study_a.run_study_a(
                    model=model,
                    data_dir=args.data_dir or "data/openr1_psy_splits",
                    max_samples=args.max_samples,
                    output_dir=args.output_dir,
                    model_name=args.model,
                    generate_only=args.generate_only,
                    from_cache=args.from_cache,
                    cache_out=args.cache_out,
                )
                results["A"] = result

            elif study == "B":
                result = study_b.run_study_b(
                    model=model,
                    data_dir=args.data_dir or "data/openr1_psy_splits",
                    max_samples=args.max_samples,
                    output_dir=args.output_dir,
                    model_name=args.model,
                    generate_only=args.study_b_generate_only,
                    from_cache=args.study_b_from_cache,
                    cache_out=args.study_b_cache_out,
                )
                results["B"] = result

            elif study == "C":
                result = study_c.run_study_c(
                    model=model,
                    data_dir=args.data_dir or "data/openr1_psy_splits",
                    max_cases=args.max_cases,
                    output_dir=args.output_dir,
                    model_name=args.model,
                    generate_only=args.study_c_generate_only,
                    from_cache=args.study_c_from_cache,
                    cache_out=args.study_c_cache_out,
                )
                results["C"] = result

            logger.info(f"Study {study} completed successfully")

        except Exception as e:
            logger.error(f"Study {study} failed: {e}", exc_info=True)
            results[study] = None

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Evaluation Summary")
    logger.info("=" * 80)

    for study, result in results.items():
        if result is None:
            logger.info(f"Study {study}: FAILED")
        else:
            logger.info(f"Study {study}: COMPLETED")
            if hasattr(result, "n_samples"):
                logger.info(f"  Samples evaluated: {result.n_samples}")
            elif hasattr(result, "n_cases"):
                logger.info(f"  Cases evaluated: {result.n_cases}")

    logger.info("")
    logger.info(f"Results saved to: {args.output_dir}/{args.model}/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()



