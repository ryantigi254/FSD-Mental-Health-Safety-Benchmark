"""
Local LM Studio runner for DeepSeek-R1-Distill-Qwen-14B (GGUF).

LM Studio applies the chat template internally for GGUF models. This runner
just sends OpenAI-style chat messages to /v1/chat/completions and captures
the resulting text. If LM Studio returns a separate 'reasoning' field, our
shared lmstudio_client will wrap it into <think>...</think> automatically.

Recommended GGUF source (example):
- https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF
"""

import os
import re
from typing import Tuple

from .base import ModelRunner, GenerationConfig
from .lmstudio_client import chat_completion


def extract_answer_and_reasoning(full_response: str) -> Tuple[str, str]:
    """
    Parse a response into (answer, reasoning) without rewriting model content.
    We only *extract* when <think> blocks exist; otherwise we fall back to heuristics.
    """
    text = (full_response or "").strip()
    think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        # Everything after the closing tag becomes the "answer" portion.
        end_tag_pos = text.find("</", think_match.end())
        if end_tag_pos != -1:
            closing_tag_end = text.find(">", end_tag_pos)
            if closing_tag_end != -1:
                answer = text[closing_tag_end + 1 :].strip()
            else:
                answer = text
        else:
            answer = text
        return answer, reasoning

    parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", text)
    if len(parts) >= 2:
        return parts[1].strip(), parts[0].strip()

    return text, text


class DeepSeekR1LMStudioRunner(ModelRunner):
    """
    Local LM Studio inference for DeepSeek-R1-Distill-Qwen-14B-GGUF.
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1-distill-qwen-14b",
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig = None,
    ):
        # Allow overriding the LM Studio API Identifier without editing code.
        # Example:
        #   $Env:LMSTUDIO_DEEPSEEK_R1_MODEL="deepseek-r1-distill-qwen-14b"
        model_name = os.getenv("LMSTUDIO_DEEPSEEK_R1_MODEL", model_name)
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
            ),
        )
        self.api_base = api_base

    def generate(self, prompt: str, mode: str = "default") -> str:
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]
        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=600,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        return extract_answer_and_reasoning(full_response)


