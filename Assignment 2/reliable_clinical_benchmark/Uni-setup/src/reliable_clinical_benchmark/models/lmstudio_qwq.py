"""
Local LM Studio runner for QwQ-32B-GGUF.

Uses the shared lmstudio_client to talk to LM Studio's /v1/chat/completions.
Model name is pinned to the LM Studio label to keep runs reproducible.
"""

import os
import re
from typing import Tuple, List, Dict
import logging

from .base import ModelRunner, GenerationConfig
from .lmstudio_client import chat_completion

logger = logging.getLogger(__name__)


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
        # Allow overriding the LM Studio API Identifier without editing code.
        # Example:
        #   $Env:LMSTUDIO_QWQ_MODEL="qwq-32b@q4_k_m"
        model_name = os.getenv("LMSTUDIO_QWQ_MODEL", model_name)
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
            timeout=None,  # No timeout - allow long generations
        )

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        """
        Generate response from chat history using LM Studio chat completion API.
        
        Properly handles multi-turn conversations with rolling context.
        """
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_messages.append({"role": "system", "content": content})
            elif role == "user":
                if msg == messages[-1] and mode != "default":
                    formatted_content = self._format_prompt(content, mode)
                    formatted_messages.append({"role": "user", "content": formatted_content})
                else:
                    formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
        
        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """
        Extract reasoning from <think> blocks when available.
        """
        full_response = self.generate(prompt, mode="cot")
        return extract_answer_and_reasoning(full_response)

