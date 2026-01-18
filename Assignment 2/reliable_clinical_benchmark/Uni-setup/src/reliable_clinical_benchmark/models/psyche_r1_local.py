"""
Local runner for Psyche-R1 using Hugging Face transformers (no LM Studio).

Loads the model from a local directory (default: Uni-setup/models/Psyche-R1)
and runs generation using the tokenizer chat template.
"""

import re
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base import ModelRunner, GenerationConfig


DEFAULT_TOP_K = 20
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 4096

PSYCHE_R1_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


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
                max_tokens=DEFAULT_MAX_NEW_TOKENS,
            ),
        )

        # Try fast tokenizer first, fall back to slow if corrupted
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, local_files_only=local_files_only
            )
        except Exception as e:
            # If fast tokenizer fails (e.g., corrupted tokenizer.json), try slow tokenizer
            import warnings
            warnings.warn(
                f"Fast tokenizer failed ({e}), falling back to slow tokenizer. "
                "This may be due to a corrupted tokenizer.json file."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False, local_files_only=local_files_only
            )
        self.tokenizer.chat_template = PSYCHE_R1_CHAT_TEMPLATE
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        self.model.eval()

    def _format_prompt(self, prompt: str, mode: str) -> str:
        """
        Psyche-R1 is typically prompted to wrap its reasoning in <think> tags.
        We still normalise outputs for the benchmark parser afterwards.
        """
        base = super()._format_prompt(prompt, mode)
        if mode == "cot":
            return (
                f"{base}\n\n"
                "Wrap your reasoning in <think>...</think> tags.\n"
                "After </think>, provide only the final diagnosis."
            )
        return base

    def _build_inputs(self, prompt: str, mode: str = "default") -> Dict[str, torch.Tensor]:
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(mode == "cot"),
            )
        except TypeError:
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

    def _strip_role_markers(self, text: str) -> str:
        """
        Defensive cleanup: some models may emit chat role markers or headers.
        We only want assistant-visible content.
        """
        t = (text or "").strip()
        # Common Qwen chat artifacts (sometimes survive decoding depending on tokenizer settings)
        t = t.replace("<|im_end|>", "").strip()
        if "<|im_start|>assistant" in t:
            t = t.split("<|im_start|>assistant", 1)[1].strip()
        return t.strip()

    def _extract_reasoning_and_answer(self, text: str) -> Tuple[str, str]:
        """
        Extract (reasoning, answer) from model text.

        Preference order:
        1) <think> / <redacted_reasoning> blocks
        2) REASONING:/DIAGNOSIS: sections
        3) Reasoning: ... then Diagnosis/Answer/Conclusion:
        4) Fallback: use the whole text for both
        """
        t = self._strip_role_markers(text)

        think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
        think_match = re.search(think_pattern, t, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
            answer = t[think_match.end() :].strip()
            return reasoning, answer.strip()

        open_think_pattern = r"<(?:redacted_reasoning|think)>\s*"
        open_think_match = re.search(open_think_pattern, t)
        if open_think_match and "</think>" not in t and "</redacted_reasoning>" not in t:
            reasoning = t[open_think_match.end() :].strip()
            return reasoning, ""

        lower = t.lower()
        if "reasoning:" in lower and "diagnosis:" in lower:
            r_start = lower.find("reasoning:") + len("reasoning:")
            d_start = lower.find("diagnosis:")
            if d_start > r_start:
                reasoning = t[r_start:d_start].strip()
                answer = t[d_start + len("diagnosis:") :].strip()
                return reasoning, answer

        # Generic fallback split
        parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", t)
        if len(parts) >= 2:
            reasoning = parts[0].strip()
            answer = parts[1].strip()
            return reasoning, answer

        return t, t

    def generate(self, prompt: str, mode: str = "default") -> str:
        encoded = self._build_inputs(prompt, mode)
        input_token_count = int(encoded["input_ids"].shape[-1])

        offload_dir = Path("offload/psyche_r1")
        offload_dir.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
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
        # Minimal post-processing: keep raw model output, only strip obvious chat artefacts.
        return self._strip_role_markers(output)

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        # Use the same normalised CoT format, then parse it deterministically.
        full_response = self.generate(prompt, mode="cot")
        reasoning, answer = self._extract_reasoning_and_answer(full_response)
        return answer.strip(), reasoning.strip()


