"""Factory for creating model runners."""

from typing import Optional
from .base import ModelRunner, GenerationConfig
from .psyllm import PsyLLMRunner
from .qwq import QwQRunner
from .deepseek_r1 import DeepSeekR1Runner
from .qwen3 import Qwen3Runner
from .gpt_oss import GPTOSSRunner
import logging

logger = logging.getLogger(__name__)


def get_model_runner(
    model_id: str, config: Optional[GenerationConfig] = None
) -> ModelRunner:
    """
    Get a model runner instance by model ID.

    Args:
        model_id: Model identifier ('psyllm', 'qwq', 'deepseek_r1', 'gpt_oss', 'qwen3')
        config: Optional generation configuration

    Returns:
        ModelRunner instance

    Raises:
        ValueError: If model_id is not recognised
    """
    model_id_lower = model_id.lower()

    if model_id_lower == "psyllm":
        return PsyLLMRunner(config=config)
    elif model_id_lower == "qwq" or model_id_lower == "qwq-32b":
        return QwQRunner(config=config)
    elif model_id_lower == "deepseek_r1" or model_id_lower == "deepseek-r1-32b":
        return DeepSeekR1Runner(config=config)
    elif model_id_lower == "gpt_oss" or model_id_lower == "gpt-oss-120b":
        return GPTOSSRunner(config=config)
    elif model_id_lower == "qwen3" or model_id_lower == "qwen3-8b":
        return Qwen3Runner(config=config)
    else:
        raise ValueError(
            f"Unknown model ID: {model_id}. "
            f"Supported models: psyllm, qwq, deepseek_r1, gpt_oss, qwen3"
        )

