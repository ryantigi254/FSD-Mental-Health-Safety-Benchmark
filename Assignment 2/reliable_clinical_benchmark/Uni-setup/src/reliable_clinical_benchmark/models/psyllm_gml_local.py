"""
Local runner for GMLHUHE/PsyLLM using Hugging Face transformers (no LM Studio).

This is the PsyLLM model from:
- https://huggingface.co/GMLHUHE/PsyLLM

It is an ~8B Qwen3-family model (BF16) and should fit in full precision on a 24GB GPU.

The model card shows Qwen3-style usage with enable_thinking and <think> blocks.
If the tokenizer repo does not ship a chat template, we inject a Qwen3-compatible
template so that apply_chat_template(enable_thinking=...) works consistently.
"""

import re
from typing import Tuple, Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base import ModelRunner, GenerationConfig


DEFAULT_TOP_K = 20

# Qwen3-style chat template (provided by you) to preserve <think> blocks and support enable_thinking.
PSYLLM_QWEN3_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for forward_message in messages %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- set message = messages[index] %}
    {%- set current_content = message.content if message.content is not none else '' %}
    {%- set tool_start = '<tool_response>' %}
    {%- set tool_start_length = tool_start|length %}
    {%- set start_of_message = current_content[:tool_start_length] %}
    {%- set tool_end = '</tool_response>' %}
    {%- set tool_end_length = tool_end|length %}
    {%- set start_pos = (current_content|length) - tool_end_length %}
    {%- if start_pos < 0 %}
        {%- set start_pos = 0 %}
    {%- endif %}
    {%- set end_of_message = current_content[start_pos:] %}
    {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = (message.content.split('</think>')|last).lstrip('\n') %}
                {%- set reasoning_content = (message.content.split('</think>')|first).rstrip('\n') %}
                {%- set reasoning_content = (reasoning_content.split('<think>')|last).lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
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
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""


class PsyLLMGMLLocalRunner(ModelRunner):
    """
    HF-local inference for GMLHUHE/PsyLLM via transformers.
    Returns raw assistant output (minimal stripping) for objective logging.
    """

    def __init__(
        self,
        model_name: str = "GMLHUHE/PsyLLM",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[GenerationConfig] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        force_chat_template: bool = True,
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
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

        if force_chat_template:
            self.tokenizer.chat_template = PSYLLM_QWEN3_CHAT_TEMPLATE

        model_config = AutoConfig.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            config=model_config,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def _build_inputs(self, prompt: str, mode: str = "default") -> Dict[str, torch.Tensor]:
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]

        prompt_text: str
        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs: Dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
            # Mirror the model card: enable_thinking for CoT.
            if mode == "cot":
                kwargs["enable_thinking"] = True
            try:
                prompt_text = self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                prompt_text = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            prompt_text = formatted_prompt

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
        return t, t

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
        return self._strip_role_markers(output)

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        reasoning, answer = self._extract_reasoning_and_answer(full_response)
        return answer.strip(), reasoning.strip()

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        """
        Generate response from chat history using transformers chat template.
        
        Properly handles multi-turn conversations with rolling context.
        """
        # Apply mode formatting to the last user message if needed
        formatted_messages = messages.copy()
        if formatted_messages and formatted_messages[-1].get("role") == "user" and mode != "default":
            last_msg = formatted_messages[-1]
            formatted_content = self._format_prompt(last_msg["content"], mode)
            formatted_messages[-1] = {"role": "user", "content": formatted_content}
        
        # Build inputs using chat template
        prompt_text: str
        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs: Dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
            if mode == "cot":
                kwargs["enable_thinking"] = True
            try:
                prompt_text = self.tokenizer.apply_chat_template(formatted_messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                prompt_text = self.tokenizer.apply_chat_template(formatted_messages, **kwargs)
        else:
            # Fallback: convert to string
            prompt_parts = []
            for msg in formatted_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_text = "\n".join(prompt_parts)
        
        # Tokenize and generate
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
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
        return self._strip_role_markers(output)


