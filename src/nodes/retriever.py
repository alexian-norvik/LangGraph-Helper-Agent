"""Document retrieval node."""

import threading

from langchain_core.documents import Document
from loguru import logger

from src.common.constants import MAX_TOTAL_DOCS, TOP_K_RESULTS
from src.data_prep.vectorstore import load_vectorstore
from src.state import AgentState

# Thread-safe cache for the vectorstore
_vectorstore_cache = None
_vectorstore_lock = threading.Lock()

# Source priority mapping based on query type
SOURCE_PRIORITY = {
    "langgraph": ["langgraph_full", "langgraph", "langchain_full", "langchain"],
    "langchain": ["langchain_full", "langchain", "langgraph_full", "langgraph"],
    "code_example": ["langgraph_full", "langchain_full", "langgraph", "langchain"],
    "general": ["langgraph_full", "langchain_full", "langgraph", "langchain"],
}


def get_vectorstore():
    """Get the vector store, loading it if necessary (thread-safe)."""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        with _vectorstore_lock:
            # Double-check locking pattern
            if _vectorstore_cache is None:
                _vectorstore_cache = load_vectorstore()
    return _vectorstore_cache


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string.

    Args:
        docs: List of retrieved documents

    Returns:
        Formatted context string
    """
    if not docs:
        return "No relevant documentation found."

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


def prioritize_docs(docs: list[Document], query_type: str) -> list[Document]:
    """Reorder documents based on source priority for the query type.

    Args:
        docs: List of retrieved documents
        query_type: Type of query (langgraph, langchain, code_example, general)

    Returns:
        Reordered list of documents
    """
    priority = SOURCE_PRIORITY.get(query_type, SOURCE_PRIORITY["general"])

    def get_priority(doc: Document) -> int:
        source = doc.metadata.get("source", "unknown")
        try:
            return priority.index(source)
        except ValueError:
            return len(priority)  # Unknown sources go last

    return sorted(docs, key=get_priority)


def multi_query_search(
    vectorstore, queries: list[str], k_per_query: int, max_total: int = MAX_TOTAL_DOCS
) -> list[Document]:
    """Search with multiple queries and deduplicate results.

    Args:
        vectorstore: The vector store to search
        queries: List of search queries
        k_per_query: Number of results per query
        max_total: Maximum total documents to return (prevents unbounded growth)

    Returns:
        Deduplicated list of documents, capped at max_total
    """
    seen_content = set()
    all_docs = []

    for query in queries:
        docs = vectorstore.similarity_search(query, k=k_per_query)
        for doc in docs:
            # Use first 200 chars as dedup key
            content_key = doc.page_content[:200]
            if content_key not in seen_content:
                seen_content.add(content_key)
                all_docs.append(doc)

                # Stop early if we've hit the cap
                if len(all_docs) >= max_total:
                    return all_docs

    return all_docs


def retriever(state: AgentState) -> dict:
    """Retrieve relevant documents from the vector store.

    Args:
        state: Current agent state with the user's query

    Returns:
        Updated state with retrieved_docs and context
    """
    query = state["query"]
    query_type = state.get("query_type", "general")
    web_results = state.get("web_results", [])

    vectorstore = get_vectorstore()

    if vectorstore is None:
        context = "Vector store not available. Please run: python scripts/prepare_data.py"
        return {"retrieved_docs": [], "context": context}

    # Build multiple search queries for better coverage
    # Always include Python-focused queries since we're a Python project
    search_queries = [
        query,
        f"Python {query}",
    ]

    if query_type == "langgraph":
        search_queries.extend(
            [
                f"LangGraph Python {query}",
                f"from langgraph {query}",
                f"builder.compile {query}",
                f"StateGraph {query}",
            ]
        )
    elif query_type == "langchain":
        search_queries.extend(
            [
                f"LangChain Python {query}",
                f"from langchain {query}",
            ]
        )
    elif query_type == "code_example":
        search_queries.extend(
            [
                f"Python code example {query}",
                f"def {query}",
            ]
        )

    # Multi-query search with deduplication
    k_per_query = TOP_K_RESULTS
    docs = multi_query_search(vectorstore, search_queries, k_per_query)

    # Prioritize docs based on query type
    docs = prioritize_docs(docs, query_type)

    # Keep only top K after prioritization
    docs = docs[:TOP_K_RESULTS]

    logger.debug(f"Query type: {query_type}, queries: {len(search_queries)}")
    logger.debug(f"Retrieved {len(docs)} documents (prioritized for {query_type})")
    for i, doc in enumerate(docs):
        logger.debug(
            f"Doc {i + 1} [{doc.metadata.get('source', 'unknown')}]: {doc.page_content[:100]}..."
        )

    # Combine with web results if available
    context_parts = []

    if web_results:
        context_parts.append("## Web Search Results\n" + "\n\n".join(web_results))

    if docs:
        context_parts.append("## Documentation\n" + format_docs(docs))

    context = "\n\n".join(context_parts) if context_parts else "No relevant information found."

    return {"retrieved_docs": docs, "context": context}
