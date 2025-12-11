"""
Local runner for Psyche-R1 using Hugging Face transformers (no LM Studio).

Loads the model from a local directory (default: Uni-setup/models/Psyche-R1)
and runs generation using the tokenizer chat template.
"""

import re
from typing import Tuple, Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base import ModelRunner, GenerationConfig


DEFAULT_TOP_K = 20
DEFAULT_REPETITION_PENALTY = 1.05


class PsycheR1LocalRunner(ModelRunner):
    """
    Local inference for Psyche-R1 via transformers.
    """

    def __init__(
        self,
        model_name: str = "models/Psyche-R1",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[GenerationConfig] = None,
        local_files_only: bool = True,
    ):
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=1e-5,
                top_p=0.8,
                max_tokens=1024,
            ),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=local_files_only
        )
        model_config = AutoConfig.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            config=model_config,
            local_files_only=local_files_only,
        )

    def _build_inputs(self, prompt: str, mode: str = "default") -> Dict[str, torch.Tensor]:
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
            top_k=DEFAULT_TOP_K,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_only = gen[0, input_token_count:]
        output = self.tokenizer.decode(generated_only, skip_special_tokens=True)
        return output.strip()

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
        think_match = re.search(think_pattern, full_response, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            after = full_response[think_match.end() :].strip()
            answer = after
        else:
            parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", full_response)
            if len(parts) >= 2:
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                reasoning = full_response
                answer = full_response

        return answer, reasoning


