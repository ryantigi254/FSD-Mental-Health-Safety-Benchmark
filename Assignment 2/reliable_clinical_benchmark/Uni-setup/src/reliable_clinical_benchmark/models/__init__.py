"""Model runners for different LLM backends."""

from .base import ModelRunner, GenerationConfig
from .factory import get_model_runner
from .psyllm import PsyLLMRunner
from .qwq import QwQRunner
from .deepseek_r1 import DeepSeekR1Runner
from .qwen3 import Qwen3Runner
from .gpt_oss import GPTOSSRunner
from .lmstudio_qwq import QwQLMStudioRunner
from .lmstudio_gpt_oss import GPTOSSLMStudioRunner
from .lmstudio_qwen3 import Qwen3LMStudioRunner
from .piaget_local import Piaget8BLocalRunner
from .psyche_r1_local import PsycheR1LocalRunner
from .psych_qwen_local import PsychQwen32BLocalRunner
from .psyllm_local import PsyLLMLocalRunner

__all__ = [
    "ModelRunner",
    "GenerationConfig",
    "get_model_runner",
    "PsyLLMRunner",
    "QwQRunner",
    "QwQLMStudioRunner",
    "DeepSeekR1Runner",
    "Qwen3Runner",
    "Qwen3LMStudioRunner",
    "GPTOSSRunner",
    "GPTOSSLMStudioRunner",
    "Piaget8BLocalRunner",
    "PsycheR1LocalRunner",
    "PsychQwen32BLocalRunner",
    "PsyLLMLocalRunner",
]

