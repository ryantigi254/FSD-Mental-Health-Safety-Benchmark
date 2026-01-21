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
        
        # Try GPU generation first
        try:
            gen = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except RuntimeError as e:
            # Check if it's a CUDA OOM error
            error_str = str(e).lower()
            if "cuda" in error_str and ("out of memory" in error_str or "oom" in error_str):
                logger.warning(f"GPU OOM error: {error_str[:200]}")
                
                # Clear GPU cache aggressively before retry
                if torch.cuda.is_available():
                    logger.info("Clearing GPU cache before retry...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(0)
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Retry with progressively reduced tokens
                for reduced_tokens in [self.config.max_tokens // 2, self.config.max_tokens // 4, 512, 256]:
                    try:
                        logger.info(f"Retrying GPU generation with max_new_tokens={reduced_tokens}")
                        gen = self.model.generate(
                            **encoded,
                            max_new_tokens=reduced_tokens,
                            do_sample=True,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                        logger.info(f"GPU generation succeeded with reduced max_new_tokens={reduced_tokens}")
                        break
                    except RuntimeError as retry_error:
                        error_str_retry = str(retry_error).lower()
                        if "cuda" in error_str_retry and ("out of memory" in error_str_retry or "oom" in error_str_retry):
                            if reduced_tokens == 256:
                                # All GPU retries failed - try CPU fallback
                                logger.warning(
                                    f"GPU generation failed even with minimal tokens ({reduced_tokens}). "
                                    f"Attempting CPU fallback (this will be slow but should work with available RAM)"
                                )
                                
                                # Clear GPU cache before CPU fallback
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                    torch.cuda.reset_peak_memory_stats(0)
                                    import gc
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                
                                # Move model and inputs to CPU
                                try:
                                    cpu_model = self.model.cpu()
                                    encoded_cpu = {k: v.cpu() for k, v in encoded.items()}
                                    
                                    logger.info(f"Attempting CPU generation with max_new_tokens={reduced_tokens}")
                                    gen = cpu_model.generate(
                                        **encoded_cpu,
                                        max_new_tokens=reduced_tokens,
                                        do_sample=True,
                                        temperature=self.config.temperature,
                                        top_p=self.config.top_p,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                    )
                                    
                                    # Move model back to GPU for next generation
                                    self.model = cpu_model.to(next(self.model.parameters()).device if torch.cuda.is_available() else "cpu")
                                    
                                    logger.info("CPU generation succeeded")
                                    break
                                except Exception as cpu_error:
                                    logger.error(f"CPU generation also failed: {cpu_error}")
                                    raise retry_error from cpu_error
                            continue
                        else:
                            raise retry_error
                else:
                    raise e
            else:
                raise
        
        generated_only = gen[0, input_token_count:]
        output = self.tokenizer.decode(generated_only, skip_special_tokens=True)
        
        # Clear GPU cache after each generation to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
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


