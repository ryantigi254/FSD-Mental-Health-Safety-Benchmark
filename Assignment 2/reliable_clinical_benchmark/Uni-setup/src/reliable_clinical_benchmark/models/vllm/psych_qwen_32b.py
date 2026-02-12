"""
vLLM-backed runner for Psych_Qwen_32B (Compumacy/Psych_Qwen_32B).

Talks to a vLLM OpenAI-compatible server on /v1/chat/completions, e.g.:

    conda activate mh-llm-hf-vllm-env
    python -m vllm.entrypoints.openai.api_server ^
        --model Compumacy/Psych_Qwen_32B ^
        --download-dir ./models/vllm ^
        --host 0.0.0.0 ^
        --port 8104 ^
        --gpu-memory-utilization 0.9 ^
        --max-num-seqs 8 ^
        --enforce-eager ^
        --max-model-len 4096
"""

from __future__ import annotations

import os
from typing import Tuple
import logging

from ..base import ModelRunner, GenerationConfig
from ..lmstudio_client import chat_completion

logger = logging.getLogger(__name__)


class PsychQwen32BVLLMRunner(ModelRunner):
    """
    vLLM OpenAI-compatible inference for Psych_Qwen_32B.
    """

    def __init__(
        self,
        model_name: str = "Compumacy/Psych_Qwen_32B",
        api_base: str | None = None,
        config: GenerationConfig | None = None,
    ):
        model_name = os.getenv("VLLM_PSYCH_QWEN_MODEL", model_name)
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=2048,
            ),
        )
        default_api_base = "http://127.0.0.1:8104/v1"
        self.api_base = os.getenv("VLLM_PSYCH_QWEN_API_BASE", api_base or default_api_base)

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
            timeout=None,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full = self.generate(prompt, mode="cot")
        return full, full

