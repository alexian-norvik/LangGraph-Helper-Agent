"""Configuration management for the LangGraph Helper Agent."""

import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from src.common.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOC_URLS,
    MAX_SEARCH_RESULTS,
    TOP_K_RESULTS,
)
from src.common.llm_constants import EMBEDDING_MODEL

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

# Agent configuration (thread-safe)
_mode_lock = threading.Lock()
_agent_mode = os.getenv("AGENT_MODE", "offline").lower()

# Re-export constants for backward compatibility
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "VECTORSTORE_DIR",
    "DOC_URLS",
    "DOC_FILES",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K_RESULTS",
    "MAX_SEARCH_RESULTS",
    "get_mode",
    "set_mode",
]


def get_mode() -> str:
    """Get the current agent mode (thread-safe)."""
    with _mode_lock:
        return _agent_mode


def set_mode(mode: str) -> None:
    """Set the agent mode dynamically (thread-safe)."""
    global _agent_mode
    if mode.lower() not in ("offline", "online"):
        raise ValueError("Mode must be 'offline' or 'online'")
    with _mode_lock:
        _agent_mode = mode.lower()
