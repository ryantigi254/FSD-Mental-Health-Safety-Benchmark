"""
Local LM Studio runner for QwQ-32B-GGUF.

Uses the shared lmstudio_client to talk to LM Studio's /v1/chat/completions.
Model name is pinned to the LM Studio label to keep runs reproducible.
"""

import re
from typing import Tuple
import logging

from .base import ModelRunner, GenerationConfig
from .lmstudio_client import chat_completion

logger = logging.getLogger(__name__)


class QwQLMStudioRunner(ModelRunner):
    """
    Local LM Studio inference for QwQ-32B-GGUF.
    """

    def __init__(
        self,
        model_name: str = "QwQ-32B-GGUF",
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig = None,
    ):
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
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
        """
        Extract reasoning from <think> blocks when available.
        """
        full_response = self.generate(prompt, mode="cot")
        think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
        think_match = re.search(think_pattern, full_response, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            end_tag_pos = full_response.find("</", think_match.end())
            if end_tag_pos != -1:
                closing_tag_end = full_response.find(">", end_tag_pos)
                if closing_tag_end != -1:
                    answer = full_response[closing_tag_end + 1 :].strip()
                else:
                    answer = full_response
            else:
                answer = full_response
        else:
            parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", full_response)
            if len(parts) >= 2:
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning = full_response
                answer = full_response

        return answer, reasoning

