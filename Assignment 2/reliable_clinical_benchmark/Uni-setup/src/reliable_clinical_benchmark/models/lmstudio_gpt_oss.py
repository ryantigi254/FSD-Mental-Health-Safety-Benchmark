"""
Local LM Studio runner for GPT-OSS-20B.

Uses the shared lmstudio_client to talk to LM Studio's /v1/chat/completions.
Model name is pinned to the LM Studio label to keep runs reproducible.
"""

import re
from typing import Tuple, List, Dict
import logging
import requests

from .base import ModelRunner, GenerationConfig
from .lmstudio_client import chat_completion

logger = logging.getLogger(__name__)

DEFAULT_GPT_OSS_MODEL_ID = "gpt-oss-20b"


def _resolve_lmstudio_model_name(api_base: str, configured_model_name: str) -> str:
    """
    Resolve GPT-OSS model ID from LM Studio /v1/models.

    LM Studio model identifiers sometimes include provider prefixes
    (e.g. "openai_gpt-oss-20b"). This resolver keeps compatibility with
    both old and new IDs.
    """
    candidate_name = (configured_model_name or DEFAULT_GPT_OSS_MODEL_ID).strip()
    if not candidate_name:
        candidate_name = DEFAULT_GPT_OSS_MODEL_ID

    endpoint = f"{api_base.rstrip('/')}/models"
    try:
        response = requests.get(endpoint, timeout=(5, 20))
        response.raise_for_status()
        payload = response.json()
        model_entries = payload.get("data", []) if isinstance(payload, dict) else []
        available_model_ids = [
            entry.get("id", "").strip()
            for entry in model_entries
            if isinstance(entry, dict) and entry.get("id")
        ]
    except Exception as request_error:
        logger.warning(
            "Could not resolve LM Studio GPT-OSS model id from %s (%s). Using configured id '%s'.",
            endpoint,
            request_error,
            candidate_name,
        )
        return candidate_name

    if not available_model_ids:
        logger.warning(
            "LM Studio returned no models from %s. Using configured id '%s'.",
            endpoint,
            candidate_name,
        )
        return candidate_name

    if candidate_name in available_model_ids:
        return candidate_name

    candidate_name_lower = candidate_name.lower()
    suffix_matches = [
        model_id
        for model_id in available_model_ids
        if model_id.lower().endswith(candidate_name_lower)
    ]
    if suffix_matches:
        resolved_model_id = suffix_matches[0]
        logger.info(
            "Resolved LM Studio model id '%s' -> '%s' via suffix match.",
            candidate_name,
            resolved_model_id,
        )
        return resolved_model_id

    contains_gpt_oss = [
        model_id
        for model_id in available_model_ids
        if "gpt-oss" in model_id.lower()
    ]
    if contains_gpt_oss:
        resolved_model_id = contains_gpt_oss[0]
        logger.info(
            "Resolved LM Studio model id '%s' -> '%s' via GPT-OSS fallback.",
            candidate_name,
            resolved_model_id,
        )
        return resolved_model_id

    logger.warning(
        "Configured model id '%s' not found in LM Studio models %s. Using configured id as-is.",
        candidate_name,
        available_model_ids,
    )
    return candidate_name


class GPTOSSLMStudioRunner(ModelRunner):
    """
    Local LM Studio inference for GPT-OSS-20B.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GPT_OSS_MODEL_ID,
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig = None,
    ):
        resolved_model_name = _resolve_lmstudio_model_name(api_base, model_name)
        super().__init__(
            resolved_model_name,
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=2048,
            ),
        )
        self.api_base = api_base

    def generate(self, prompt: str, mode: str = "default") -> str:
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]
        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,  # No timeout - allow long generations
        )

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        """
        Generate response from chat history using LM Studio chat completion API.
        
        Properly handles multi-turn conversations with rolling context.
        """
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_messages.append({"role": "system", "content": content})
            elif role == "user":
                if msg == messages[-1] and mode != "default":
                    formatted_content = self._format_prompt(content, mode)
                    formatted_messages.append({"role": "user", "content": formatted_content})
                else:
                    formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
        
        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """
        Extract reasoning from <think> blocks when available.
        """
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
