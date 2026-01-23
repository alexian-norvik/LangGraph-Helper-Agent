"""LLM-related constants for the LangGraph Helper Agent."""

# LLM configuration is now in config.yaml (uses lc-openrouter-ollama-client)
# Supports both Ollama (local) and OpenRouter (cloud) providers

# Embedding model (Ollama - local, no rate limits)
# snowflake-arctic-embed2: 1024 dims, 8K context, high quality
# Run: ollama pull snowflake-arctic-embed2
EMBEDDING_MODEL = "snowflake-arctic-embed2"

# Default LLM parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_TIMEOUT = 30  # Request timeout in seconds
OPENROUTER_DEFAULT_MAX_RETRIES = 3  # Maximum retry attempts

# Ollama configuration
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_TIMEOUT = 60  # Request timeout in seconds
