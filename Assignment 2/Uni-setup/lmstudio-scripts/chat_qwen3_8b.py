import argparse
import json
import os
from pathlib import Path

import requests

DEFAULT_BASE = "http://127.0.0.1:1234"
DEFAULT_URL = f"{DEFAULT_BASE}/v1/chat/completions"
DEFAULT_SYSTEM = (
    "You are Qwen3-8B running through LM Studio. Respond with balanced, clear, "
    "and safe guidance for wellbeing conversations."
)
MODEL_ID = "qwen3-8b"


def call_lmstudio(prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    url = os.environ.get("LM_STUDIO_URL", DEFAULT_URL)
    api_key = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Unexpected response: {json.dumps(data, indent=2)}")

    return data["choices"][0]["message"]["content"].strip()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query LM Studio with the local Qwen3-8B GGUF model."
    )
    parser.add_argument(
        "prompt",
        help="User prompt to send to the model. Wrap in quotes for multi-line text.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="Override the default system prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature passed to LM Studio.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate.",
    )

    args = parser.parse_args()

    print(f"Calling LM Studio at {os.environ.get('LM_STUDIO_URL', DEFAULT_URL)}")
    print(f"Model id: {MODEL_ID}\n")

    reply = call_lmstudio(
        prompt=args.prompt,
        system_prompt=args.system,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("===== MODEL REPLY =====")
    print(reply)

if __name__ == "__main__":
    main()
