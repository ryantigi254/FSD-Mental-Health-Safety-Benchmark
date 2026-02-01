"""
Smoke test for Study B Multi-Turn with a dummy model or small run.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.pipelines.study_b import run_study_b
from datetime import datetime

if __name__ == "__main__":
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    config = GenerationConfig(max_tokens=64) # Small max tokens for smoke test
    
    # Use a common model-id for testing (e.g., qwen3_lmstudio if available)
    model_id = "qwen3_lmstudio" 
    
    print(f"Starting Multi-Turn Smoke Test for {model_id}...")
    
    cache_out = f"results/{model_id}/study_b_multi_turn_generations.smoke-{run_id}.jsonl"
    
    run_study_b(
        model=get_model_runner(model_id, config),
        data_dir="data/openr1_psy_splits",
        max_samples=1, # Only 1 case
        output_dir="results",
        model_name=model_id,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
        do_single_turn=False,
        do_multi_turn=True,
    )
    
    print(f"[ok] Multi-turn smoke test complete. Output: {cache_out}")
