"""
Shared LM Studio HTTP client for all locally-hosted models.
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


