"""Piaget-8B model runner via Hugging Face Inference API.

This runner is an optional extension: it allows evaluating the
`gustavecortal/Piaget-8B` model on the same benchmark harness as the
other reasoning models. By default it uses the Hugging Face Inference
API with an API key provided in HUGGINGFACE_API_KEY.
"""

from .remote_api import RemoteAPIRunner, GenerationConfig


class Piaget8BRunner(RemoteAPIRunner):
    """Piaget-8B inference via Hugging Face Inference API."""

    def __init__(self, config: GenerationConfig = None):
        super().__init__(
            model_name="gustavecortal/Piaget-8B",
            api_endpoint="https://api-inference.huggingface.co/models/gustavecortal/Piaget-8B",
            api_key_env="HUGGINGFACE_API_KEY",
            config=config,
        )



