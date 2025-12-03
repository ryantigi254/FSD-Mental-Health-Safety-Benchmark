"""
Base class for remote API-based model runners.
"""

import os
import requests
import re
from typing import Tuple
from .base import ModelRunner, GenerationConfig
import logging

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()
logger = logging.getLogger(__name__)


class RemoteAPIRunner(ModelRunner):
    """Base class for remote API-based models."""

    def __init__(
        self,
        model_name: str,
        api_endpoint: str,
        api_key_env: str,
        config: GenerationConfig = None,
    ):
        super().__init__(model_name, config)
        self.api_endpoint = api_endpoint
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            logger.warning(
                f"API key not found in environment variable {api_key_env}. "
                f"Set it in .env file before running evaluations."
            )

    def generate(self, prompt: str, mode: str = "default") -> str:
        """Generate response via remote API."""
        formatted_prompt = self._format_prompt(prompt, mode)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {self.model_name}: {e}")
            raise

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """Extract reasoning from response."""
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


