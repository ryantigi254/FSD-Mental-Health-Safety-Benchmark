"""
Shared LM Studio HTTP client for all locally-hosted models.

This module provides a single, unified interface for communicating with
LM Studio's local server API. All models running via LM Studio (PsyLLM,
and potentially others like QwQ-32B, DeepSeek-R1-32B if loaded locally)
should use this client to ensure consistent error handling and timeout management.

For this dissertation, PsyLLM is the primary model evaluated, running
locally via LM Studio. Other models (QwQ, DeepSeek-R1, etc.) are configured
as remote API runners for spec completeness, but the actual evaluation
focuses on local PsyLLM inference.
"""

import requests
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def chat_completion(
    api_base: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    timeout: int = 60,
) -> str:
    """
    Single shared helper for LM Studio /v1/chat/completions endpoint.

    This function handles all HTTP communication with LM Studio, providing:
    - Consistent error handling (timeouts, connection errors, HTTP errors)
    - Standardised request/response parsing
    - Logging for debugging

    Args:
        api_base: Base URL for LM Studio API (e.g., "http://localhost:1234/v1")
        model: Model name/identifier as recognised by LM Studio
        messages: List of message dicts with "role" and "content" keys
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        timeout: Request timeout in seconds (default: 60)

    Returns:
        Generated text content from the model

    Raises:
        TimeoutError: If request exceeds timeout
        requests.exceptions.RequestException: For other HTTP/connection errors

    Example:
        >>> messages = [{"role": "user", "content": "What is depression?"}]
        >>> response = chat_completion(
        ...     api_base="http://localhost:1234/v1",
        ...     model="PsyLLM-8B",
        ...     messages=messages,
        ...     temperature=0.7,
        ...     max_tokens=512,
        ...     top_p=0.9
        ... )
    """
    endpoint = f"{api_base}/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()

        # Extract content from OpenAI-compatible response format
        content = result["choices"][0]["message"]["content"]
        return content

    except requests.exceptions.Timeout:
        logger.error(f"LM Studio request timed out after {timeout}s for model {model}")
        raise TimeoutError(f"Generation timed out after {timeout}s")

    except requests.exceptions.HTTPError as e:
        logger.error(f"LM Studio HTTP error for model {model}: {e}")
        raise

    except requests.exceptions.RequestException as e:
        logger.error(f"LM Studio connection error for model {model}: {e}")
        raise

    except (KeyError, IndexError) as e:
        logger.error(f"LM Studio response parsing error for model {model}: {e}")
        logger.error(f"Response structure: {result if 'result' in locals() else 'N/A'}")
        raise ValueError(f"Unexpected response format from LM Studio: {e}")

