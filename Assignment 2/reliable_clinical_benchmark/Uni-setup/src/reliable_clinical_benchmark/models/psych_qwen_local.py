"""
Local runner for Psych_Qwen_32B using Hugging Face transformers (no LM Studio).

Designed for 24GB VRAM setups:
- Default quantization is 4-bit NF4 (bitsandbytes) for weights
- Optional CPU offload via device_map="auto" + max_memory

Supports loading either:
- A fully merged model checkpoint, OR
- A PEFT/LoRA adapter repo (loads base model then applies adapter)
"""

import re
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base import ModelRunner, GenerationConfig

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 20
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 4096

# Qwen3-style chat template that preserves <think> blocks and supports enable_thinking.
PSYCH_QWEN_CHAT_TEMPLATE = r"""{%- if tools %}
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


class PsychQwen32BLocalRunner(ModelRunner):
    """
    Local inference for Psych_Qwen_32B (Qwen3 architecture) via transformers.
    """

    def __init__(
        self,
        model_name: str = "models/Psych_Qwen_32B",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        quantization: Optional[str] = "4bit",
        max_memory: Optional[Dict] = None,
        offload_folder: str = "offload/psych_qwen_32b",
        config: Optional[GenerationConfig] = None,
        local_files_only: bool = False,
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
        # Ensure the chat template matches Qwen3-style thinking/non-thinking separation.
        self.tokenizer.chat_template = PSYCH_QWEN_CHAT_TEMPLATE
        model_config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)

        # Quantization + offload notes:
        # - 32B in bf16 won't fit 24GB VRAM. 8-bit usually still too big (~32GB weights).
        # - 4-bit (NF4) is typically required; any remainder can be offloaded to CPU RAM.
        # - On Windows, bitsandbytes support can be finicky. We keep it optional and
        #   raise a clear error if requested but unavailable.
        # - Quantization requires CUDA; if no GPU is available, skip quantization.
        q = (quantization or "").lower().strip()
        quantization_config = None
        has_cuda = torch.cuda.is_available()
        
        if q in ("4bit", "4-bit", "bnb4", "bnb_4bit", "nf4"):
            if not has_cuda:
                raise RuntimeError(
                    "4-bit quantization requires CUDA/GPU, but no GPU is available. "
                    "Either use a GPU-enabled system or set quantization='none' for CPU-only inference."
                )
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "4-bit quantization requested but BitsAndBytesConfig is unavailable. "
                    "Install bitsandbytes (and a compatible CUDA build), or use WSL2/Linux."
                ) from e
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        elif q in ("8bit", "8-bit", "bnb8", "bnb_8bit"):
            if not has_cuda:
                raise RuntimeError(
                    "8-bit quantization requires CUDA/GPU, but no GPU is available. "
                    "Either use a GPU-enabled system or set quantization='none' for CPU-only inference."
                )
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "8-bit quantization requested but BitsAndBytesConfig is unavailable. "
                    "Install bitsandbytes (and a compatible CUDA build), or use WSL2/Linux."
                ) from e
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif q in ("", "none", "no"):
            quantization_config = None
        else:
            raise ValueError("quantization must be one of: None, '8bit', '4bit'")

        # Sensible defaults for 24GB VRAM + 64GB RAM (tweak as needed).
        # 'max_memory' is honored by transformers/accelerate when device_map is used.
        # Only set max_memory if CUDA is available (like other local models).
        if max_memory is None and device_map == "auto" and torch.cuda.is_available():
            max_memory = {0: "22GiB", "cpu": "48GiB"}

        # Detect PEFT adapter repos: if adapter_config.json exists locally (or in cache),
        # load base_model_name_or_path from it and apply adapter weights.
        adapter_base: Optional[str] = None
        adapter_config_path: Optional[Path] = None
        try:
            resolved_dir = Path(model_name)
            if resolved_dir.exists() and resolved_dir.is_dir():
                candidate = resolved_dir / "adapter_config.json"
                if candidate.exists():
                    adapter_config_path = candidate
        except Exception:
            adapter_config_path = None

        if adapter_config_path is not None:
            try:
                data = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                adapter_base = data.get("base_model_name_or_path")
            except Exception:
                adapter_base = None

        if adapter_base:
            base_model = AutoModelForCausalLM.from_pretrained(
                adapter_base,
                device_map=device_map,
                dtype=dtype,
                config=AutoConfig.from_pretrained(adapter_base, local_files_only=local_files_only),
                quantization_config=quantization_config,
                max_memory=max_memory,
                offload_folder=offload_folder,
                local_files_only=local_files_only,
            )
            try:
                from peft import PeftModel
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "This checkpoint looks like a PEFT/LoRA adapter (adapter_config.json present), "
                    "but peft is not installed. Install peft to load it."
                ) from e
            self.model = PeftModel.from_pretrained(base_model, model_name, local_files_only=local_files_only)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                dtype=dtype,
                config=model_config,
                quantization_config=quantization_config,
                max_memory=max_memory,
                offload_folder=offload_folder,
                local_files_only=local_files_only,
            )

        self.model.eval()
        # Store whether quantization is enabled (quantized models require CUDA, can't use CPU fallback)
        self._is_quantized = quantization_config is not None
        # Store init params for CPU fallback reload
        self._model_name = model_name
        self._device_map = device_map
        self._dtype = dtype
        self._max_memory = max_memory
        self._offload_folder = offload_folder
        self._local_files_only = local_files_only
        self._model_config = model_config
        self._cpu_model = None  # Lazy-loaded CPU model for fallback

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

    def _get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage in GiB."""
        if not torch.cuda.is_available():
            return {"total": 0.0, "allocated": 0.0, "reserved": 0.0, "free": 0.0}
        
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free = total - reserved
        
        return {
            "total": total,
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
        }

    def _estimate_generation_memory(self, input_token_count: int, max_new_tokens: int) -> float:
        """Rough estimate of memory needed for generation in GiB."""
        # Very rough estimate: ~2 bytes per token for KV cache + activations
        # This is conservative and model-dependent
        estimated_gb = (input_token_count + max_new_tokens) * 2 / (1024**3)
        return estimated_gb

    def _load_cpu_model(self):
        """Lazy-load a 4-bit quantized CPU model for fallback generation."""
        if self._cpu_model is not None:
            return self._cpu_model
        
        logger.warning(
            "Loading 4-bit quantized model on CPU for fallback generation. "
            "This will be slow but should work with available RAM (~40 GB free). "
            "4-bit quantization reduces memory to ~16 GB for a 32B model."
        )
        
        # Try to use 4-bit quantization on CPU first (lowest memory usage)
        # Note: bitsandbytes may not fully support CPU, but we try it first
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            logger.info("Attempting to load with 4-bit quantization on CPU...")
            self._cpu_model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                device_map="cpu",  # Force CPU
                quantization_config=quantization_config,
                config=self._model_config,
                local_files_only=self._local_files_only,
            )
            logger.info("CPU model loaded successfully with 4-bit quantization")
        except Exception as e:
            logger.warning(
                f"4-bit quantization on CPU failed ({e}). "
                "Falling back to 8-bit quantization..."
            )
            try:
                # Try 8-bit as fallback
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                logger.info("Attempting to load with 8-bit quantization on CPU...")
                self._cpu_model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    device_map="cpu",  # Force CPU
                    quantization_config=quantization_config,
                    config=self._model_config,
                    local_files_only=self._local_files_only,
                )
                logger.info("CPU model loaded successfully with 8-bit quantization")
            except Exception as e2:
                logger.warning(
                    f"8-bit quantization on CPU also failed ({e2}). "
                    "Falling back to float16 (will use more memory but should still work)."
                )
                # Final fallback to float16 if quantization doesn't work on CPU
                self._cpu_model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    device_map="cpu",  # Force CPU
                    dtype=torch.float16,  # Use float16 as fallback
                    config=self._model_config,
                    quantization_config=None,
                    local_files_only=self._local_files_only,
                )
                logger.info("CPU model loaded successfully with float16 (fallback)")
        
        self._cpu_model.eval()
        return self._cpu_model

    def generate(self, prompt: str, mode: str = "default") -> str:
        encoded = self._build_inputs(prompt, mode)
        input_token_count = int(encoded["input_ids"].shape[-1])
        
        # Check memory before generation
        mem_info = self._get_memory_info()
        estimated_needed = self._estimate_generation_memory(input_token_count, self.config.max_tokens)
        
        logger.info(
            f"Memory before generation: {mem_info['free']:.2f} GiB free / {mem_info['total']:.2f} GiB total "
            f"(allocated: {mem_info['allocated']:.2f} GiB, reserved: {mem_info['reserved']:.2f} GiB). "
            f"Estimated needed: ~{estimated_needed:.2f} GiB for {self.config.max_tokens} new tokens"
        )
        
        # Dynamically reduce max_new_tokens if memory is tight
        max_new_tokens = self.config.max_tokens
        if mem_info["free"] < estimated_needed * 1.5:  # Need 1.5x buffer
            # Reduce max_new_tokens proportionally
            reduction_factor = (mem_info["free"] / (estimated_needed * 1.5))
            max_new_tokens = max(256, int(self.config.max_tokens * reduction_factor))
            logger.warning(
                f"Memory tight ({mem_info['free']:.2f} GiB free < {estimated_needed * 1.5:.2f} GiB needed). "
                f"Reducing max_new_tokens from {self.config.max_tokens} to {max_new_tokens}"
            )

        # Try GPU generation first
        try:
            gen = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=DEFAULT_TOP_K,
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        except RuntimeError as e:
            # Check if it's a CUDA OOM error
            error_str = str(e).lower()
            if "cuda" in error_str and ("out of memory" in error_str or "oom" in error_str):
                logger.warning(f"GPU OOM error: {error_str[:200]}")
                
                # Since user has plenty of RAM (~40 GB free), try CPU fallback immediately
                # This is faster than multiple GPU retries that are likely to fail
                logger.info("GPU OOM detected. With sufficient RAM available, attempting CPU fallback immediately")
                
                # For quantized models, reload without quantization on CPU
                # For non-quantized models, just move to CPU
                if self._is_quantized:
                    logger.info("Reloading model on CPU without quantization (this will use RAM but should work)")
                    cpu_model = self._load_cpu_model()
                else:
                    cpu_model = self.model
                
                encoded_cpu = {k: v.cpu() for k, v in encoded.items()}
                try:
                    logger.info(f"Attempting CPU generation with max_new_tokens={max_new_tokens} (this will be slow but should work with available RAM)")
                    gen = cpu_model.generate(
                        **encoded_cpu,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=DEFAULT_TOP_K,
                        repetition_penalty=DEFAULT_REPETITION_PENALTY,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    logger.info("CPU generation succeeded")
                except Exception as cpu_error:
                    logger.warning(f"CPU generation failed: {cpu_error}. Trying with reduced max_new_tokens...")
                    # If CPU also fails, try with reduced tokens
                    for reduced_tokens in [max_new_tokens // 2, max_new_tokens // 4, 512, 256]:
                        try:
                            logger.info(f"Retrying CPU generation with max_new_tokens={reduced_tokens}")
                            gen = cpu_model.generate(
                                **encoded_cpu,
                                max_new_tokens=reduced_tokens,
                                do_sample=True,
                                temperature=self.config.temperature,
                                top_p=self.config.top_p,
                                top_k=DEFAULT_TOP_K,
                                repetition_penalty=DEFAULT_REPETITION_PENALTY,
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.pad_token_id,
                            )
                            logger.info(f"CPU generation succeeded with reduced max_new_tokens={reduced_tokens}")
                            break
                        except Exception as cpu_error2:
                            if reduced_tokens == 256:
                                logger.error(f"CPU generation failed even with minimal tokens: {cpu_error2}")
                                raise cpu_error from cpu_error2
                            continue
                    else:
                        raise cpu_error
            else:
                raise

        generated_only = gen[0, input_token_count:]
        output = self.tokenizer.decode(generated_only, skip_special_tokens=True)
        # Return raw assistant text (minimal stripping) for objective analysis.
        return self._strip_role_markers(output)

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        reasoning, answer = self._extract_reasoning_and_answer(full_response)
        return answer.strip(), reasoning.strip()


