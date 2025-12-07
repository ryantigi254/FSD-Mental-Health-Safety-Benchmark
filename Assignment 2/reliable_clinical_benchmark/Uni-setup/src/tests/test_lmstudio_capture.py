"""
Smoke test for LM Studio capture on gpt-oss-20b: fetch raw JSON, flatten
chat template parts, and ensure we record text even if LM Studio returns
chunked content (e.g., text/tool blocks or <think> traces).

Run:
    PYTHONPATH=src python src/tests/test_lmstudio_capture.py
"""

import json
from typing import Any, Dict, List, Tuple

import requests


def _flatten_content(content: Any) -> str:
    """
    Handle LM Studio / OpenAI-style mixed content:
    - str -> returned as-is
    - list of blocks -> join text fields (covers text, reasoning, or generic)
    - fallback -> repr for inspection
    """
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
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    timeout: int = 60,
) -> Tuple[Dict[str, Any], Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "tool_choice": "none",  # avoid empty tool messages for this smoke check
    }
    endpoint = f"{api_base}/chat/completions"
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return data, content


def main() -> None:
    api_base = "http://127.0.0.1:1234/v1"
    model = "gpt-oss-20b"  # matches LM Studio load; see HF card for template
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
            raw, content = _chat_completion_raw(
                api_base=api_base,
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
            )
            flattened = _flatten_content(content)
        except Exception as e:
            raw = {"error": str(e)}
            flattened = f"[ERROR] {e}"

        print(f"--- Prompt {i} ---")
        print(p)
        print("raw_choice:")
        print(json.dumps(raw.get("choices", raw), indent=2)[:2000])  # trim for readability
        print("flattened_content:")
        print(flattened)
        print()


if __name__ == "__main__":
    main()