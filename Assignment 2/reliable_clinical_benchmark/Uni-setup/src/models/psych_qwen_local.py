"""
Local runner for Psych_Qwen_32B using Hugging Face transformers (no LM Studio).

Loads from: Uni-setup/models/Psych_Qwen_32B
"""

import re
from typing import Tuple, Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base import ModelRunner, GenerationConfig


DEFAULT_TOP_K = 20


class PsychQwen32BLocalRunner(ModelRunner):
    def __init__(
        self,
        model_name: str = "models/Psych_Qwen_32B",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[GenerationConfig] = None,
        local_files_only: bool = True,
    ):
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.6,
                top_p=0.95,
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
            enable_thinking=(mode == "cot"),
        )
        tokenized = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return {k: v.to(self.model.device) for k, v in tokenized.items()}

    def _strip_role_markers(self, text: str) -> str:
        t = (text or "").strip()
        t = t.replace("<|im_end|>", "").strip()
        if "<|im_start|>assistant" in t:
            t = t.split("<|im_start|>assistant", 1)[1].strip()
        return t.strip()

    def _extract_reasoning_and_answer(self, text: str) -> Tuple[str, str]:
        t = self._strip_role_markers(text)

        think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
        think_match = re.search(think_pattern, t, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
            answer = t[think_match.end() :].strip()
            return reasoning, answer.strip()

        lower = t.lower()
        if "reasoning:" in lower and "diagnosis:" in lower:
            r_start = lower.find("reasoning:") + len("reasoning:")
            d_start = lower.find("diagnosis:")
            if d_start > r_start:
                reasoning = t[r_start:d_start].strip()
                answer = t[d_start + len("diagnosis:") :].strip()
                return reasoning, answer

        parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", t)
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()

        return t, t

    def _normalize_for_mode(self, text: str, mode: str) -> str:
        t = self._strip_role_markers(text)

        if mode == "cot":
            reasoning, answer = self._extract_reasoning_and_answer(t)
            return f"REASONING:\n{reasoning.strip()}\n\nDIAGNOSIS:\n{answer.strip()}".strip()

        if mode == "direct":
            _, answer = self._extract_reasoning_and_answer(t)
            ans = (answer or "").strip()
            for line in ans.splitlines():
                line = line.strip()
                if line:
                    return line
            return ans

        return t.strip()

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
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_only = gen[0, input_token_count:]
        output = self.tokenizer.decode(generated_only, skip_special_tokens=True)
        return self._normalize_for_mode(output, mode)

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        reasoning, answer = self._extract_reasoning_and_answer(full_response)
        return answer.strip(), reasoning.strip()


