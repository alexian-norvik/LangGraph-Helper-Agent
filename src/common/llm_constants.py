"""LLM-related constants for the LangGraph Helper Agent."""

# Model configuration
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

# Temperature settings
CLASSIFICATION_TEMPERATURE = 0  # Deterministic for classification
GENERATION_TEMPERATURE = 0.3  # Slight creativity for responses
