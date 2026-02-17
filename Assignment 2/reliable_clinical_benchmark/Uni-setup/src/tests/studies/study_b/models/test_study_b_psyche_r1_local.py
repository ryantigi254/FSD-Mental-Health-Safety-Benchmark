"""
Smoke test for Study B with Psyche-R1 (HF local).

Run (PowerShell):
    cd Uni-setup
    python src/tests/studies/study_b/models/test_study_b_psyche_r1_local.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.pipelines.study_b import run_study_b
from datetime import datetime

if __name__ == "__main__":
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    config = GenerationConfig(max_tokens=512)
    runner = get_model_runner("psyche_r1_local", config)
    cache_out = f"results/psyche_r1_local/study_b_generations.smoke-{run_id}.jsonl"
    
    run_study_b(
        model=runner,
        data_dir="data/openr1_psy_splits",
        max_samples=1,
        output_dir="results",
        model_name="psyche_r1_local",
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
    )
    print(f"[ok] wrote {cache_out}")

