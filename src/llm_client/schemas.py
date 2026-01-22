"""Pydantic schemas for LLM client configuration validation."""

from typing import Literal

from pydantic import BaseModel, Field


class LLMParameters(BaseModel):
    """Standard LLM generation parameters."""

    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens in response")


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Model identifier")


class OpenRouterSettings(BaseModel):
    """OpenRouter-specific settings."""

    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")


class OllamaSettings(BaseModel):
    """Ollama-specific settings."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")


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
