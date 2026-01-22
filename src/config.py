"""Configuration management for the LangGraph Helper Agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.common.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOC_URLS,
    MAX_SEARCH_RESULTS,
    TOP_K_RESULTS,
)
from src.common.llm_constants import EMBEDDING_MODEL, LLM_MODEL

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Document file paths (derived from DOC_URLS keys)
DOC_FILES = {
    "langgraph": DATA_DIR / "langgraph-llms.txt",
    "langgraph_full": DATA_DIR / "langgraph-llms-full.txt",
    "langchain": DATA_DIR / "langchain-llms.txt",
    "langchain_full": DATA_DIR / "langchain-llms-full.txt",
}

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Agent configuration
AGENT_MODE = os.getenv("AGENT_MODE", "offline").lower()
ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "false").lower() == "true"

# Re-export constants for backward compatibility
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "VECTORSTORE_DIR",
    "DOC_URLS",
    "DOC_FILES",
    "GOOGLE_API_KEY",
    "AGENT_MODE",
    "ENABLE_MEMORY",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K_RESULTS",
    "MAX_SEARCH_RESULTS",
    "validate_config",
    "get_mode",
    "set_mode",
]


def validate_config() -> None:
    """Validate that required configuration is present."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required. "
            "Get your key at: https://aistudio.google.com/app/apikey"
        )


def get_mode() -> str:
    """Get the current agent mode."""
    return AGENT_MODE


def set_mode(mode: str) -> None:
    """Set the agent mode dynamically."""
    global AGENT_MODE
    if mode.lower() not in ("offline", "online"):
        raise ValueError("Mode must be 'offline' or 'online'")
    AGENT_MODE = mode.lower()
