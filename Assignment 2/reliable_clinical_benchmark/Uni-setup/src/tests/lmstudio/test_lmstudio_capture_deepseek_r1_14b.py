"""
Smoke test for LM Studio capture on DeepSeek-R1-Distill-Qwen-14B-GGUF.

This hits LM Studio directly and prints the raw JSON choice + flattened content.
Use it to verify that:
- Requests reach LM Studio
- You get output back
- If LM Studio provides reasoning separately, it is present (either in content blocks
  or as a 'reasoning' field)

Run:
    PYTHONPATH=src python src/tests/test_lmstudio_capture_deepseek_r1_14b.py --model deepseek-r1-distill-qwen-14b
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
                else:
                    parts.append(json.dumps(block))
            else:
                parts.append(str(block))
        return "".join(parts)
    return repr(content)


def _chat_completion_raw(
    api_base: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    timeout: int = 60,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "tool_choice": "none",
    }
    endpoint = f"{api_base}/chat/completions"
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content")
    return data, content, payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://127.0.0.1:1234/v1")
    p.add_argument(
        "--model",
        default=os.getenv("LMSTUDIO_DEEPSEEK_R1_MODEL", "deepseek-r1-distill-qwen-14b"),
        help="LM Studio API Identifier (or set $Env:LMSTUDIO_DEEPSEEK_R1_MODEL).",
    )
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    prompts = [
        "How many r's are in the word \"strawberry\"?",
        "Give me a one-line clinical differential for persistent low mood.",
    ]

    for i, ptxt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": ptxt}]
        try:
            raw, content, sent = _chat_completion_raw(
                api_base=args.api_base,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
            )
            flattened = _flatten_content(content)
        except Exception as e:
            raw = {"error": str(e)}
            flattened = f"[ERROR] {e}"
            sent = {}

        log = {
            "run_id": run_id,
            "prompt_idx": i,
            "prompt": ptxt,
            "request": sent,
            "response_meta": {
                "id": raw.get("id") if isinstance(raw, dict) else None,
                "created": raw.get("created") if isinstance(raw, dict) else None,
                "model": raw.get("model") if isinstance(raw, dict) else None,
                "usage": raw.get("usage") if isinstance(raw, dict) else None,
            }
            if isinstance(raw, dict)
            else {},
            "raw_choice": raw.get("choices", raw) if isinstance(raw, dict) else raw,
            "flattened_content": flattened,
        }

        print(f"--- Prompt {i} ---")
        print(json.dumps(log, indent=2)[:4000])
        print()


if __name__ == "__main__":
    main()


