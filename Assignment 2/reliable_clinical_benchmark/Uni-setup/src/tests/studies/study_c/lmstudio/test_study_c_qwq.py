"""
Smoke test for Study C with QwQ-32B (LM Studio).

Run (PowerShell):
    cd Uni-setup
    python src/tests/studies/study_c/lmstudio/test_study_c_qwq.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.pipelines.study_c import run_study_c
from datetime import datetime

if __name__ == "__main__":
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    config = GenerationConfig(max_tokens=512)
    runner = get_model_runner("qwq", config)
    cache_out = f"results/qwq/study_c_generations.smoke-{run_id}.jsonl"
    
    run_study_c(
        model=runner,
        data_dir="data/openr1_psy_splits",
        max_cases=1,
        output_dir="results",
        model_name="qwq",
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
    )
    print(f"[ok] wrote {cache_out}")

