"""Factory for creating model runners."""

from typing import Optional
from .base import ModelRunner, GenerationConfig
from .psyllm import PsyLLMRunner
from .qwq import QwQRunner
from .deepseek_r1 import DeepSeekR1Runner
from .qwen3 import Qwen3Runner
from .gpt_oss import GPTOSSRunner
from .piaget import Piaget8BRunner
from .psyche_r1 import PsycheR1Runner
from .psych_qwen import PsychQwen32BRunner
import logging

logger = logging.getLogger(__name__)


def get_model_runner(
    model_id: str, config: Optional[GenerationConfig] = None
) -> ModelRunner:
    """
    Get a model runner instance by model ID.
    """
    model_id_lower = model_id.lower()

    if model_id_lower == "psyllm":
        return PsyLLMRunner(config=config)
    elif model_id_lower in ("qwq", "qwq-32b"):
        return QwQRunner(config=config)
    elif model_id_lower in ("deepseek_r1", "deepseek-r1-32b"):
        return DeepSeekR1Runner(config=config)
    elif model_id_lower in ("gpt_oss", "gpt-oss-120b"):
        return GPTOSSRunner(config=config)
    elif model_id_lower in ("qwen3", "qwen3-8b"):
        return Qwen3Runner(config=config)
    elif model_id_lower in ("piaget", "piaget-8b"):
        return Piaget8BRunner(config=config)
    elif model_id_lower in ("psyche_r1", "psyche-r1"):
        return PsycheR1Runner(config=config)
    elif model_id_lower in ("psych_qwen", "psych_qwen_32b", "psych-qwen-32b"):
        return PsychQwen32BRunner(config=config)
    else:
        raise ValueError(
            f"Unknown model ID: {model_id}. "
            f"Supported models: psyllm, qwq, deepseek_r1, gpt_oss, qwen3, "
            f"piaget, psyche_r1, psych_qwen"
        )


