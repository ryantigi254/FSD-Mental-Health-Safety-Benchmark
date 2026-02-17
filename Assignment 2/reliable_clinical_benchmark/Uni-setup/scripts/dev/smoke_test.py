import sys
import argparse
from pathlib import Path

# Setup path to import src
uni_setup_root = Path(__file__).resolve().parent
src_dir = uni_setup_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.models.lmstudio_qwen3 import Qwen3LMStudioRunner

def main():
    print("Initializing Qwen3LMStudioRunner...")
    runner = Qwen3LMStudioRunner(
        model_name="deepseek-r1-distill-qwen-7b",
        api_base="http://127.0.0.1:1234/v1",
        config=GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        ),
    )

    print("Sending smoke test prompt...")
    try:
        response = runner.generate("Hello, are you receiving this message? Please reply with 'Yes'.", mode="direct")
        print("\n--- Model Response ---")
        print(response)
        print("----------------------")
        print("Smoke test PASSED.")
    except Exception as e:
        print(f"\nSmoke test FAILED: {e}")

if __name__ == "__main__":
    main()
