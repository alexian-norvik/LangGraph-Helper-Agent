"""LLM-related constants for the LangGraph Helper Agent."""

# LLM configuration is now in config.yaml (uses lc-openrouter-ollama-client)
# Supports both Ollama (local) and OpenRouter (cloud) providers

# Embedding model (Ollama - local, no rate limits)
# snowflake-arctic-embed2: 1024 dims, 8K context, high quality
# Run: ollama pull snowflake-arctic-embed2
EMBEDDING_MODEL = "snowflake-arctic-embed2"
