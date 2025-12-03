"""
PsyLLM model runner using LM Studio local inference.

This is the primary model runner for this dissertation. PsyLLM runs locally
via LM Studio, with all HTTP communication handled by the shared lmstudio_client.

All other model runners (QwQ, DeepSeek-R1, etc.) are configured as remote API
runners for spec completeness, but the actual evaluation focuses on local
PsyLLM inference via this runner.
"""

import re
from typing import Tuple
from .base import ModelRunner, GenerationConfig
from .lmstudio_client import chat_completion
import logging

logger = logging.getLogger(__name__)


class PsyLLMRunner(ModelRunner):
    """
    Local inference via LM Studio for PsyLLM.

    This runner uses the shared lmstudio_client for all HTTP communication,
    ensuring consistent error handling and timeout management across all
    locally-hosted models.
    """

    def __init__(
        self,
        model_name: str = "PsyLLM-8B",
        api_base: str = "http://localhost:1234/v1",
        config: GenerationConfig = None,
    ):
        super().__init__(model_name, config)
        self.api_base = api_base

    def generate(self, prompt: str, mode: str = "default") -> str:
        """
        Generate response via LM Studio API.

        Uses the shared lmstudio_client for all HTTP communication.
        """
        formatted_prompt = self._format_prompt(prompt, mode)

        messages = [{"role": "user", "content": formatted_prompt}]

        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=60,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """
        Extract reasoning from PsyLLM response.
        PsyLLM uses <think>...</think> tags for reasoning traces.
        """
        full_response = self.generate(prompt, mode="cot")

        # Extract reasoning from <think> or <think> tags
        think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
        think_match = re.search(think_pattern, full_response, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            # Extract answer after the closing tag
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
            # Fallback: split by common separators
            parts = re.split(
                r"\n(?:Diagnosis|Answer|Conclusion):\s*", full_response
            )
            if len(parts) >= 2:
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning = full_response
                answer = full_response

        return answer, reasoning

