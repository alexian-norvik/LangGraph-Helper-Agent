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
