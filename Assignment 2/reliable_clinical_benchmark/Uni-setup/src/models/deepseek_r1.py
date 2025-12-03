"""DeepSeek-R1-32B model runner via Hugging Face API."""

from .remote_api import RemoteAPIRunner, GenerationConfig


class DeepSeekR1Runner(RemoteAPIRunner):
    """DeepSeek-R1-32B inference via Hugging Face Inference API."""

    def __init__(self, config: GenerationConfig = None):
        super().__init__(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            api_endpoint="https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            api_key_env="HUGGINGFACE_API_KEY",
            config=config,
        )


