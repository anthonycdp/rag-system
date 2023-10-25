"""LLM factory for consistent initialization across modules."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.constants import DEFAULT_LLM_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


def create_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
    **extra_kwargs: Any,
) -> BaseChatModel:
    """Create a ChatOpenAI LLM instance with sensible defaults.

    Args:
        model_name: OpenAI model name.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens in response.
        api_key: OpenAI API key (falls back to environment).
        **extra_kwargs: Additional arguments passed to ChatOpenAI.

    Returns:
        Configured ChatOpenAI instance.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if api_key:
        kwargs["api_key"] = api_key

    kwargs.update(extra_kwargs)

    return ChatOpenAI(**kwargs)
