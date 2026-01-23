"""General constants for the LangGraph Helper Agent."""

# Documentation URLs (from LangGraph llms.txt overview)
DOC_URLS = {
    "langgraph": "https://langchain-ai.github.io/langgraph/llms.txt",
    "langgraph_full": "https://langchain-ai.github.io/langgraph/llms-full.txt",
    "langchain": "https://docs.langchain.com/llms.txt",
    "langchain_full": "https://docs.langchain.com/llms-full.txt",
}

# Chunking configuration
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Retrieval configuration
TOP_K_RESULTS = 8

# Web search configuration
MAX_SEARCH_RESULTS = 3

# Query classification
VALID_QUERY_TYPES = {"langgraph", "langchain", "code_example", "general"}
DEFAULT_QUERY_TYPE = "langgraph"

# Input validation limits
MAX_QUERY_LENGTH = 2000  # Maximum characters in a query
MAX_CHAT_HISTORY = 20  # Maximum number of messages to keep in history
MAX_TOTAL_DOCS = 15  # Maximum total documents after multi-query search

# Prompt injection protection - patterns that suggest malicious input
SUSPICIOUS_PATTERNS = [
    "ignore all previous",
    "ignore previous instructions",
    "disregard all previous",
    "forget your instructions",
    "new instructions:",
    "system prompt:",
    "you are now",
    "act as if",
    "pretend you are",
    "roleplay as",
    "jailbreak",
    "bypass your",
]
