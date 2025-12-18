"""
Local runner for gustavecortal/Piaget-8B using Hugging Face transformers.

Loads the model into local memory (no LM Studio) and runs generation with
chat template + enable_thinking to preserve <think> traces.
"""

import re
from typing import Tuple
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base import ModelRunner, GenerationConfig

logger = logging.getLogger(__name__)


class Piaget8BLocalRunner(ModelRunner):
    """
    Local inference for Piaget-8B via transformers.
    """

    def __init__(
        self,
        model_name: str = "models/Piaget-8B",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        config: GenerationConfig = None,
    ):
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            ),
        )
        logger.info(f"Loading {model_name} locally (device_map={device_map})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
            config=config,
        )

    def _build_inputs(self, prompt: str, mode: str = "default"):
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        tokenized = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return {k: v.to(self.model.device) for k, v in tokenized.items()}

    def generate(self, prompt: str, mode: str = "default") -> str:
        encoded = self._build_inputs(prompt, mode)
        input_token_count = int(encoded["input_ids"].shape[-1])
        gen = self.model.generate(
            **encoded,
            max_new_tokens=self.config.max_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_only = gen[0, input_token_count:]
        output = self.tokenizer.decode(generated_only, skip_special_tokens=True)
        return output.strip()

    def _strip_prompt(self, text: str) -> str:
        # Remove echoed prompt if present
        split = text.split("<|im_start|>assistant", 1)
        if len(split) == 2:
            return split[1].replace("<|im_end|>", "").strip()
        return text.strip()

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
        think_match = re.search(think_pattern, full_response, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            end_tag_pos = full_response.find("</", think_match.end())
            if end_tag_pos != -1:
                closing_tag_end = full_response.find(">", end_tag_pos)
                if closing_tag_end != -1:
                    answer = full_response[closing_tag_end + 1 :].strip()
                else:
                    answer = full_response
            else:
                answer = full_response
        else:
            parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", full_response)
            if len(parts) >= 2:
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning = full_response
                answer = full_response

        return answer, reasoning


