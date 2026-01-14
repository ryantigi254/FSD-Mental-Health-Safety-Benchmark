"""Factory for creating model runners."""

from typing import Optional
from .base import ModelRunner, GenerationConfig
from .psyllm import PsyLLMRunner
from .psyllm_local import PsyLLMLocalRunner
from .psyllm_gml_local import PsyLLMGMLLocalRunner
from .deepseek_r1 import DeepSeekR1Runner
from .lmstudio_deepseek_r1 import DeepSeekR1LMStudioRunner
from .qwen3 import Qwen3Runner
from .gpt_oss import GPTOSSRunner
from .piaget import Piaget8BRunner
from .piaget_local import Piaget8BLocalRunner
from .psyche_r1 import PsycheR1Runner
from .psyche_r1_local import PsycheR1LocalRunner
from .psych_qwen import PsychQwen32BRunner
from .psych_qwen_local import PsychQwen32BLocalRunner
from .lmstudio_qwq import QwQLMStudioRunner
from .lmstudio_gpt_oss import GPTOSSLMStudioRunner
from .lmstudio_qwen3 import Qwen3LMStudioRunner
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
    elif model_id_lower in ("psyllm_local", "psyllm-8b-local", "psyllm-local-hf"):
        return PsyLLMLocalRunner(config=config)
    elif model_id_lower in ("psyllm_gml_local", "psyllm-gml-local", "psyllm-gmlhuhe-local", "gmlhuhe_psyllm_local"):
        return PsyLLMGMLLocalRunner(config=config)
    elif model_id_lower in ("qwq", "qwq-32b", "qwq_lmstudio", "qwq-lmstudio", "qwq-32b-lmstudio"):
        return QwQLMStudioRunner(config=config)
    elif model_id_lower in ("deepseek_r1", "deepseek-r1-32b"):
        return DeepSeekR1Runner(config=config)
    elif model_id_lower in ("deepseek_r1_lmstudio", "deepseek-r1-lmstudio", "deepseek-r1-14b-lmstudio"):
        return DeepSeekR1LMStudioRunner(config=config)
    elif model_id_lower in ("gpt_oss", "gpt_oss_lmstudio", "gpt-oss-lmstudio", "gpt-oss-20b"):
        # Default GPT-OSS runs are expected to use LM Studio (local OpenAI-compatible server).
        return GPTOSSLMStudioRunner(config=config)
    elif model_id_lower in ("gpt_oss_remote", "gpt_oss_120b_remote", "gpt-oss-120b"):
        # Placeholder remote runner (api.example.com) retained for completeness.
        return GPTOSSRunner(config=config)
    elif model_id_lower in ("qwen3", "qwen3-8b"):
        return Qwen3Runner(config=config)
    elif model_id_lower in ("qwen3_lmstudio", "qwen3-lmstudio", "qwen3-8b-lmstudio"):
        return Qwen3LMStudioRunner(config=config)
    elif model_id_lower in ("piaget", "piaget-8b"):
        return Piaget8BRunner(config=config)
    elif model_id_lower in ("piaget_local", "piaget-8b-local", "piaget8b-local"):
        return Piaget8BLocalRunner(config=config)
    elif model_id_lower in ("psyche_r1", "psyche-r1"):
        return PsycheR1Runner(config=config)
    elif model_id_lower in ("psyche_r1_local", "psyche-r1-local", "psyche-r1-local-hf"):
        return PsycheR1LocalRunner(config=config)
    elif model_id_lower in ("psych_qwen", "psych_qwen_32b", "psych-qwen-32b"):
        return PsychQwen32BRunner(config=config)
    elif model_id_lower in ("psych_qwen_local", "psych-qwen-32b-local", "psych-qwen-local-hf"):
        return PsychQwen32BLocalRunner(config=config)
    else:
        raise ValueError(
            f"Unknown model ID: {model_id}. "
            f"Supported models: psyllm, qwq, deepseek_r1, gpt_oss, qwen3, "
            f"piaget, psyche_r1, psych_qwen"
        )

