"""Psych_Qwen_32B model runner via Hugging Face Inference API.

This runner targets `Compumacy/Psych_Qwen_32B`, a 32B Qwen-based
psychological reasoning model. Because of its size, weights are
expected to be hosted remotely or managed via an external inference
server; this runner simply provides a consistent ModelRunner interface.
"""

from .remote_api import RemoteAPIRunner, GenerationConfig


class PsychQwen32BRunner(RemoteAPIRunner):
    """Psych_Qwen_32B inference via Hugging Face Inference API."""

    def __init__(self, config: GenerationConfig = None):
        super().__init__(
            model_name="Compumacy/Psych_Qwen_32B",
            api_endpoint="https://api-inference.huggingface.co/models/Compumacy/Psych_Qwen_32B",
            api_key_env="HUGGINGFACE_API_KEY",
            config=config,
        )



