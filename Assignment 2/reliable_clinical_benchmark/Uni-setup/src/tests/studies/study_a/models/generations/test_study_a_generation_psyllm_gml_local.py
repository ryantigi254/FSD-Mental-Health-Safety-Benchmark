"""
Smoke test for Study A generation with PsyLLM (HF local, GMLHUHE/PsyLLM).

Run (PowerShell):
    cd Uni-setup
    python src/tests/studies/study_a/models/generations/test_study_a_generation_psyllm_gml_local.py
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
    runner = get_model_runner("psyllm_gml_local", config)
    cache_out = f"results/psyllm_gml_local/study_a_generations.smoke-{run_id}.jsonl"
    
    run_study_a(
        model=runner,
        data_dir="data/study_a",
        max_samples=2,
        output_dir="results",
        model_name="psyllm_gml_local",
        generate_only=True,
        cache_out=cache_out,
    )
    print(f"[ok] wrote {cache_out}")
