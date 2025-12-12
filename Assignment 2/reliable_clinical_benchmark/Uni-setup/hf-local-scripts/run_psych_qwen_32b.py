import argparse

from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner
from reliable_clinical_benchmark.models.base import GenerationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--quantization",
        default="4bit",
        help="Quantization mode (e.g. 4bit, 8bit, none). Default: 4bit",
    )
    args = parser.parse_args()

    runner = PsychQwen32BLocalRunner(
        model_name="../models/Psych_Qwen_32B",
        quantization=args.quantization,
        config=GenerationConfig(
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_new_tokens,
        ),
    )
    print(runner.generate(args.prompt, mode="cot"))


if __name__ == "__main__":
    main()

