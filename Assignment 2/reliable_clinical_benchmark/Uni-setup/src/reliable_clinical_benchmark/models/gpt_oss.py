"""GPT-OSS-120B model runner (placeholder)."""

from .remote_api import RemoteAPIRunner, GenerationConfig
import logging

logger = logging.getLogger(__name__)


class GPTOSSRunner(RemoteAPIRunner):
    """GPT-OSS-120B inference (placeholder - update with actual endpoint)."""

    def __init__(self, config: GenerationConfig = None):
        logger.warning(
            "GPT-OSS-120B runner is a placeholder. "
            "Update src/models/gpt_oss.py with actual API endpoint before use."
        )
        super().__init__(
            model_name="gpt-oss/120B",
            api_endpoint="https://api.example.com/v1/chat/completions",  # PLACEHOLDER
            api_key_env="GPT_OSS_API_KEY",
            config=config,
        )

