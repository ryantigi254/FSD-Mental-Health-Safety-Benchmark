"""
Smoke test for Study A generation with DeepSeek-R1 (LM Studio distill).

Run (PowerShell):
    cd Uni-setup
    python src/tests/studies/study_a/lmstudio/generations/test_study_a_generation_deepseek_r1_lmstudio.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.pipelines.study_a import run_study_a
from datetime import datetime

if __name__ == "__main__":
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    config = GenerationConfig(max_tokens=512)
    runner = get_model_runner("deepseek_r1_lmstudio", config)
    cache_out = f"results/deepseek_r1_lmstudio/study_a_generations.smoke-{run_id}.jsonl"
    
    run_study_a(
        model=runner,
        data_dir="data/study_a",
        max_samples=2,
        output_dir="results",
        model_name="deepseek_r1_lmstudio",
        generate_only=True,
        cache_out=cache_out,
    )
    print(f"[ok] wrote {cache_out}")

