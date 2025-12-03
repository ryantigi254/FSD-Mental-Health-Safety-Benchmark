"""Piaget-8B model runner using local weights.

This runner loads `gustavecortal/Piaget-8B` from the local
`psy-llm-local/models/Piaget-8B` folder (kept out of git) and exposes a
standard ModelRunner interface.
"""

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ModelRunner, GenerationConfig
import logging

logger = logging.getLogger(__name__)


class Piaget8BRunner(ModelRunner):
    """Piaget-8B inference from local weights."""

    def __init__(self, config: Optional[GenerationConfig] = None, model_dir: Optional[Path] = None):
        super().__init__("Piaget-8B", config)

        root = Path(__file__).resolve().parents[4]
        default_dir = root / "psy-llm-local" / "models" / "Piaget-8B"
        self.model_dir = Path(model_dir) if model_dir is not None else default_dir

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Local Piaget-8B weights not found at {self.model_dir}. "
                f"Download them into psy-llm-local/models/Piaget-8B first."
            )

        logger.info(f"Loading Piaget-8B from {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def generate(self, prompt: str, mode: str = "default") -> str:
        formatted_prompt = self._format_prompt(prompt, mode)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                top_p=self.config.top_p,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text



