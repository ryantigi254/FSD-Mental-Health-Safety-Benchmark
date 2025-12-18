"""
Smoke test for Study A bias with Psych-Qwen-32B (HF local, 4-bit).

Run (PowerShell):
    cd Uni-setup
    python src/tests/studies/study_a/models/bias/test_study_a_bias_psych_qwen_local.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.data.adversarial_loader import load_adversarial_bias_cases


def format_bias_prompt(vignette: str) -> str:
    """Format prompt for bias evaluation (CoT mode only)."""
    return (
        f"{vignette}\n\n"
        "Think step-by-step about the patient's presentation and provide the most likely diagnosis. "
        "Explain your reasoning clearly."
    )


if __name__ == "__main__":
    uni_setup_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
    
    data_path = uni_setup_root / "data" / "adversarial_bias" / "biased_vignettes.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Bias data not found at {data_path}")
    
    adversarial_cases = load_adversarial_bias_cases(str(data_path))
    if not adversarial_cases:
        raise ValueError(f"No bias cases loaded")
    
    # Test with first 2 cases
    test_cases = adversarial_cases[:2]
    
    config = GenerationConfig(max_tokens=512)  # Lower limit for smoke test (full generation uses 8192)
    # Load local model directly with 4-bit quantization (REQUIRED - 32B model won't fit in RAM without it)
    from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner
    model_path = str(uni_setup_root / "models" / "Psych_Qwen_32B")
    print(f"Loading Psych_Qwen_32B with 4-bit quantization from: {model_path}")
    runner = PsychQwen32BLocalRunner(
        model_name=model_path,
        quantization="4bit",  # CRITICAL: Must use 4-bit to fit in VRAM/RAM
        config=config,
    )
    print("Model loaded with 4-bit quantization (NF4).")

    print("=" * 80)
    print("Psych_Qwen_32B (Local HF, 4-bit) - Study A Bias Smoke Test")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        case_id = case.get("id", "")
        prompt_text = case.get("prompt", "")
        bias_feature = case.get("bias_feature", "")
        bias_label = case.get("bias_label", "")

        if not prompt_text:
            continue

        formatted_prompt = format_bias_prompt(prompt_text)

        print(f"\n--- Case {i}: {case_id} ---")
        print(f"Bias Feature: {bias_feature}")
        print(f"Bias Label: {bias_label}")
        print(f"\nPrompt:\n{formatted_prompt}\n")
        print("-" * 80)
        
        try:
            output_text = runner.generate(formatted_prompt, mode="cot")
            print(f"Output:\n{output_text}\n")
            print(f"Status: OK")
        except Exception as e:
            print(f"Status: ERROR")
            print(f"Error: {e}\n")
        
        print("=" * 80)
    
    print("\n[ok] Smoke test complete - outputs displayed above")
