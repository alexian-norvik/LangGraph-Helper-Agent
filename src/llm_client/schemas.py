"""Pydantic schemas for LLM client configuration validation."""

from typing import Literal

from pydantic import BaseModel, Field

from src.common.llm_constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_DEFAULT_TIMEOUT,
    OPENROUTER_DEFAULT_MAX_RETRIES,
    OPENROUTER_DEFAULT_TIMEOUT,
)


class LLMParameters(BaseModel):
    """Standard LLM generation parameters."""

    temperature: float = Field(
        default=DEFAULT_TEMPERATURE, ge=0.0, le=1.0, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS, gt=0, description="Maximum tokens in response"
    )


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Model identifier")


class OpenRouterSettings(BaseModel):
    """OpenRouter-specific settings."""

    timeout: int = Field(
        default=OPENROUTER_DEFAULT_TIMEOUT, gt=0, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=OPENROUTER_DEFAULT_MAX_RETRIES, ge=0, description="Maximum retry attempts"
    )


class OllamaSettings(BaseModel):
    """Ollama-specific settings."""

    base_url: str = Field(default=OLLAMA_DEFAULT_BASE_URL, description="Ollama server URL")
    timeout: int = Field(
        default=OLLAMA_DEFAULT_TIMEOUT, gt=0, description="Request timeout in seconds"
    )


class LLMConfig(BaseModel):
    """Complete LLM configuration."""

    platform: Literal["openrouter", "ollama"] = Field(..., description="LLM platform to use")
    model: ModelConfig = Field(..., description="Model configuration")
    parameters: LLMParameters = Field(
        default_factory=LLMParameters, description="Generation parameters"
    )
    openrouter: OpenRouterSettings | None = Field(
        default=None, description="OpenRouter-specific settings"
    )
    ollama: OllamaSettings | None = Field(default=None, description="Ollama-specific settings")
