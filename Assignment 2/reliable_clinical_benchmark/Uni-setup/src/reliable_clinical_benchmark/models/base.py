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

        Args:
            prompt: Input text
            mode: Generation mode ('cot', 'direct', 'summary')
                - 'cot': Request step-by-step reasoning
                - 'direct': Request immediate answer only
                - 'summary': Request brief summary

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """
        Generate response with explicit reasoning trace.

        Args:
            prompt: Input text

        Returns:
            Tuple of (answer, reasoning_trace)
        """
        pass

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        """
        Generate response from chat history (multi-turn conversations).
        
        This method handles iterative generation with rolling context. Each call
        should include the full conversation history up to that point, including
        previous assistant responses.
        
        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Roles should be "system", "user", or "assistant".
                     Example: [
                         {"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": "Hi there!"},
                         {"role": "user", "content": "How are you?"}
                     ]
            mode: Generation mode ('cot', 'direct', 'summary')
        
        Returns:
            Generated text response from the model
        
        Default implementation converts messages to a single prompt string.
        Subclasses should override this for proper chat template handling.
        """
        # Default: convert messages to a single prompt string
        # This works for simple cases but subclasses should override for proper chat templates
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts)
        return self.generate(prompt, mode=mode)

    def _format_prompt(self, prompt: str, mode: str) -> str:
        """Format prompt based on generation mode."""
        if mode == "cot":
            return (
                f"Think step-by-step about this case:\n\n{prompt}\n\n"
                "Provide your reasoning before stating your diagnosis."
            )
        elif mode == "direct":
            return f"{prompt}\n\nProvide only the diagnosis (no explanation):"
        elif mode == "summary":
            return f"Summarise the following:\n\n{prompt}"
        else:
            return prompt

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

