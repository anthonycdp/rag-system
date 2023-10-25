"""JSON extraction utilities for parsing LLM responses."""

import json
from typing import Any


def extract_json_from_response(content: str) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response.

    LLMs often wrap JSON in markdown code blocks. This function handles
    extracting the JSON from various formats.

    Args:
        content: Raw response content from LLM.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    cleaned = content.strip()

    # Try to extract from markdown code blocks
    if "```json" in cleaned:
        json_str = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        json_str = cleaned.split("```")[1].split("```")[0].strip()
    else:
        json_str = cleaned

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}") from e


def safe_extract_json(content: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Safely extract JSON without raising exceptions.

    Args:
        content: Raw response content from LLM.
        default: Default value to return if parsing fails.

    Returns:
        Parsed JSON dictionary or default value.
    """
    try:
        return extract_json_from_response(content)
    except ValueError:
        return default if default is not None else {}
