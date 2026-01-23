"""Unified client for OpenRouter and Ollama LLM providers."""

from collections.abc import Iterator

import httpx
import requests
from langchain.chat_models import init_chat_model
from loguru import logger

from src.common.llm_constants import OPENROUTER_BASE_URL
from src.general_utils.config_loader import load_yaml_config, validate_config
from src.llm_client.schemas import LLMConfig, OllamaSettings, OpenRouterSettings
from src.llm_client.utils import (
    ProviderError,
    load_env_var,
)


class UnifiedLLMClient:
    """Unified client for OpenRouter and Ollama LLM providers."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the LLM client.

        Args:
            config_path: Path to YAML configuration file
        """
        config_dict = load_yaml_config(config_path)
        self.config: LLMConfig = validate_config(config_dict.get("llm", {}), LLMConfig)
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration."""
        platform = self.config.platform

        if platform == "openrouter":
            return self._init_openrouter()
        elif platform == "ollama":
            return self._init_ollama()
        else:
            raise ValueError(f"Unknown platform: {platform}")

    def _init_openrouter(self):
        """Initialize OpenRouter LLM."""
        api_key = load_env_var("OPENROUTER_API_KEY", required=True)

        settings = self.config.openrouter or OpenRouterSettings()

        logger.info(f"Initializing OpenRouter with model: {self.config.model.name}")

        return init_chat_model(
            model=self.config.model.name,
            model_provider="openai",
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=self.config.parameters.temperature,
            max_tokens=self.config.parameters.max_tokens,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

    def _init_ollama(self):
        """Initialize Ollama LLM."""
        settings = self.config.ollama or OllamaSettings()

        logger.info(f"Initializing Ollama with model: {self.config.model.name}")

        return init_chat_model(
            model=self.config.model.name,
            model_provider="ollama",
            base_url=settings.base_url,
            temperature=self.config.parameters.temperature,
            num_predict=self.config.parameters.max_tokens,
        )

    def invoke(self, prompt: str) -> str:
        """Send a prompt and get a response.

        Args:
            prompt: The input prompt

        Returns:
            The model's response as a string

        Raises:
            ProviderError: If the LLM request fails
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except (requests.RequestException, httpx.HTTPError) as e:
            logger.error(f"HTTP error invoking LLM: {e}")
            raise ProviderError(f"Failed to invoke LLM: {e}") from e
        except (TimeoutError, ConnectionError) as e:
            logger.error(f"Connection error invoking LLM: {e}")
            raise ProviderError(f"Failed to invoke LLM: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid input to LLM: {e}")
            raise ProviderError(f"Failed to invoke LLM: {e}") from e

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream a response from the LLM.

        Args:
            prompt: The input prompt

        Yields:
            Response chunks as strings

        Raises:
            ProviderError: If the LLM streaming fails
        """
        try:
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)
        except (requests.RequestException, httpx.HTTPError) as e:
            logger.error(f"HTTP error streaming from LLM: {e}")
            raise ProviderError(f"Failed to stream from LLM: {e}") from e
        except (TimeoutError, ConnectionError) as e:
            logger.error(f"Connection error streaming from LLM: {e}")
            raise ProviderError(f"Failed to stream from LLM: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid input to LLM: {e}")
            raise ProviderError(f"Failed to stream from LLM: {e}") from e
