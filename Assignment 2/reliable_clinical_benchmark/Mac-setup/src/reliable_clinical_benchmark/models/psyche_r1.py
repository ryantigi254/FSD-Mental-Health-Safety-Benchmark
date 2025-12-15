"""Psyche-R1 model runner via Hugging Face Inference API.

Optional extension runner for `MindIntLab/Psyche-R1`, a psychological
reasoning model. Uses the Hugging Face Inference API when an API key is
available in HUGGINGFACE_API_KEY.
"""

from .remote_api import RemoteAPIRunner, GenerationConfig


class PsycheR1Runner(RemoteAPIRunner):
    """Psyche-R1 inference via Hugging Face Inference API."""

    def __init__(self, config: GenerationConfig = None):
        super().__init__(
            model_name="MindIntLab/Psyche-R1",
            api_endpoint="https://api-inference.huggingface.co/models/MindIntLab/Psyche-R1",
            api_key_env="HUGGINGFACE_API_KEY",
            config=config,
        )


