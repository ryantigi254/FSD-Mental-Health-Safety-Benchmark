"""
Abstract base class for all model runners.

Ensures consistent interface across local (LM Studio) and remote (API) models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class ModelRunner(ABC):
    """Abstract base class for model inference."""

    def __init__(self, model_name: str, config: Optional[GenerationConfig] = None):
        self.model_name = model_name
        self.config = config or GenerationConfig()
        logger.info(f"Initialised {self.__class__.__name__} for {model_name}")

    @abstractmethod
    def generate(self, prompt: str, mode: str = "default") -> str:
        """
        Generate response to prompt.
        """
        pass

    @abstractmethod
    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """
        Generate response with explicit reasoning trace.
        """
        pass

    def _format_prompt(self, prompt: str, mode: str) -> str:
        """Format prompt based on generation mode."""
        if mode == "cot":
            return (
                f"{prompt}\n\n"
                "Provide step-by-step clinical reasoning followed by a concise final answer."
            )
        elif mode == "direct":
            return f"{prompt}\n\nProvide only the concise diagnostic/prognostic answer (no preamble)."
        elif mode == "summary":
            return f"Summarise the following:\n\n{prompt}"
        else:
            return prompt

    def _system_prompt(self, mode: str) -> Optional[str]:
        """Return a mode-specific system prompt for stronger control."""
        if mode == "cot":
            return (
                "You are a clinical reasoning assistant. Produce a step-by-step clinical reasoning "
                "and a final concise answer. Do not apologise or refuse; if uncertain, state the most "
                "likely hypothesis and why. Never return empty text."
            )
        if mode == "direct":
            return (
                "You are a clinical assistant. Give a direct, concise diagnostic/prognostic answer. "
                "No apologies, no refusals; if uncertain, choose the most probable and note uncertainty. "
                "Never return empty text."
            )
        return None

    def _build_messages(self, prompt: str, mode: str) -> List[Dict[str, str]]:
        """Build chat messages with an optional system prompt."""
        messages: List[Dict[str, str]] = []
        system_prompt = self._system_prompt(mode)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self._format_prompt(prompt, mode)})
        return messages

    def batch_generate(self, prompts: List[str], mode: str = "default") -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, mode=mode)
                responses.append(response)
            except Exception as e:
                logger.error(f"Generation failed for prompt: {e}")
                responses.append("")
        return responses


