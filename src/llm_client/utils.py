"""Utility functions for LLM client."""

import os

from dotenv import load_dotenv


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    pass


class ConfigurationError(LLMClientError):
    """Configuration validation or loading errors."""

    pass


class ProviderError(LLMClientError):
    """Provider-specific errors."""

    pass


class APIKeyError(LLMClientError):
    """Missing or invalid API key."""

    pass


def load_env_var(key: str, required: bool = True) -> str | None:
    """Load environment variable with validation.

    Args:
        key: Environment variable name
        required: If True, raises error when variable is missing

    Returns:
        Environment variable value or None

    Raises:
        APIKeyError: If required variable is missing
    """
    load_dotenv()

    value = os.getenv(key)

    if required and not value:
        raise APIKeyError(f"{key} not found in environment. Please set it in your .env file.")

    return value
