"""
Smoke test for LM Studio capture: sends a few prompts and prints raw outputs.

This is a non-asserting test-style script to help verify that:
- tool_choice=none prevents empty content tool-calls
- we capture any returned <think> traces in content

Run:
    PYTHONPATH=src python src/tests/test_lmstudio_capture.py
"""

from reliable_clinical_benchmark.models.lmstudio_client import chat_completion


def main() -> None:
    api_base = "http://127.0.0.1:1234/v1"
    model = "qwen3-8b-mlx"  # adjust to the loaded LM Studio model id
    prompts = [
        "Give me a one-line clinical differential for persistent low mood.",
        "List two bullet coping tips for test anxiety.",
        "Explain why empty tool calls might appear; keep it short.",
        "Provide a brief step-by-step reasoning (<think> tags allowed) for insomnia causes.",
        "State one concise diagnosis guess for: 'throat tightness and shaking when anxious'.",
    ]

    for i, p in enumerate(prompts, 1):
        messages = [{"role": "user", "content": p}]
        try:
            out = chat_completion(
                api_base=api_base,
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
            )
        except Exception as e:
            out = f"[ERROR] {e}"
        print(f"--- Prompt {i} ---")
        print(p)
        print(">>>")
        print(out)
        print()


if __name__ == "__main__":
    main()

