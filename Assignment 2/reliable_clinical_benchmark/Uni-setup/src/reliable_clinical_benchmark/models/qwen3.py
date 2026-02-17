"""Qwen3-8B baseline model runner via Hugging Face API."""

from .remote_api import RemoteAPIRunner, GenerationConfig


class Qwen3Runner(RemoteAPIRunner):
    """Qwen3-8B baseline inference via Hugging Face Inference API."""

    def __init__(self, config: GenerationConfig = None):
        super().__init__(
            model_name="Qwen/Qwen3-8B",
            api_endpoint="https://api-inference.huggingface.co/models/Qwen/Qwen3-8B",
            api_key_env="HUGGINGFACE_API_KEY",
            config=config,
        )

