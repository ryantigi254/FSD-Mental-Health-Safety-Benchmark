"""QwQ-32B model runner via Hugging Face API."""

from .remote_api import RemoteAPIRunner, GenerationConfig


class QwQRunner(RemoteAPIRunner):
    """QwQ-32B inference via Hugging Face Inference API."""

    def __init__(self, config: GenerationConfig = None):
        super().__init__(
            model_name="Qwen/QwQ-32B-Preview",
            api_endpoint="https://api-inference.huggingface.co/models/Qwen/QwQ-32B-Preview",
            api_key_env="HUGGINGFACE_API_KEY",
            config=config,
        )

