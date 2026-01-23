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
MIN_CHUNK_SIZE = 100  # Minimum chunk size (filters out header-only chunks)

# Markdown separators for text splitting (in order of preference)
MARKDOWN_SEPARATORS = [
    "\n## ",  # H2 headers
    "\n### ",  # H3 headers
    "\n#### ",  # H4 headers
    "\n\n",  # Paragraphs
    "\n",  # Lines
    ". ",  # Sentences
    " ",  # Words
    "",  # Characters
]

# Retrieval configuration
TOP_K_RESULTS = 8

# Web search configuration
MAX_SEARCH_RESULTS = 3
DDGS_REGION = "wt-wt"  # No region bias for DuckDuckGo

# Query classification
VALID_QUERY_TYPES = {"langgraph", "langchain", "code_example", "general"}
DEFAULT_QUERY_TYPE = "langgraph"

# Input validation limits
MAX_QUERY_LENGTH = 2000  # Maximum characters in a query
MAX_CHAT_HISTORY = 20  # Maximum number of messages to keep in history
MAX_TOTAL_DOCS = 15  # Maximum total documents after multi-query search
CHAT_HISTORY_CONTEXT_LENGTH = 6  # Number of recent messages to include in prompt

# Document deduplication
DEDUP_KEY_LENGTH = 200  # Characters used as dedup key for documents
MIN_SECTION_LENGTH = 50  # Minimum section length for dedup
DEDUP_SIGNATURE_LENGTH = 200  # Characters used as signature for section dedup

# Vectorstore configuration
VECTORSTORE_BATCH_SIZE = 100  # Batch size for embedding documents

# Download configuration
DOWNLOAD_TIMEOUT = 60  # HTTP request timeout in seconds

# Preprocessing configuration
MAX_CONSECUTIVE_BLANK_LINES = 4  # Max blank lines before cleanup

# Code block patterns for preprocessing
EMPTY_CODE_BLOCK_PATTERN = r"```\w*\s*```"
WHITESPACE_CODE_BLOCK_PATTERN = r"```\w*\n\s*```"

# Navigation and boilerplate patterns to remove
NAVIGATION_REMOVAL_PATTERNS = [
    r"^\s*[\w\s]+>\s*[\w\s]+>\s*[\w\s]+\s*$",  # Breadcrumbs
    r"^\s*-\s*\[.*?\]\(#.*?\)\s*$",  # TOC links
    r"On this page\s*\n(?:\s*-\s*\[.*?\].*?\n)+",  # "On this page" sections
    r"Skip to (?:main )?content",  # "Skip to content" links
    r"(?:Previous|Next):\s*\[.*?\]\(.*?\)",  # Footer navigation
    r"\[Edit (?:this page )?on GitHub\].*",  # Edit on GitHub links
    r"Was this (?:page |)helpful\?.*",  # Was this helpful sections
    r"\[\]\(.*?\)",  # Empty markdown links
    r"(?:---\s*){2,}",  # Repeated separator lines
]

# Python code indicators for detecting Python in generic code blocks
PYTHON_CODE_INDICATORS = [
    "import ",
    "from ",
    "def ",
    "class ",
    "async def",
    "if __name__",
    '"""',
    "'''",
    "print(",
    "return ",
    "@tool",
    "@",
    "self.",
    "None",
    "True",
    "False",
]

# UI/Display configuration
HEADER_WIDTH = 60  # Width of separator lines in CLI output

# Logging level mapping
LOG_LEVEL_MAP = {
    "quiet": "ERROR",
    "normal": "WARNING",
    "verbose": "INFO",
    "debug": "DEBUG",
}

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
