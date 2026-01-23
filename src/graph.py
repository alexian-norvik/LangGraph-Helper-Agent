"""StateGraph definition for the LangGraph Helper Agent."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.common.constants import MAX_CHAT_HISTORY, MAX_QUERY_LENGTH, SUSPICIOUS_PATTERNS
from src.nodes.answer_generator import answer_generator
from src.nodes.query_classifier import query_classifier
from src.nodes.retriever import retriever
from src.nodes.web_search import web_search
from src.state import AgentState


def sanitize_query(query: str) -> str:
    """Sanitize user query for basic prompt injection protection.

    Args:
        query: The user's raw query

    Returns:
        Sanitized query

    Note:
        This provides basic protection against common prompt injection patterns.
        It logs warnings but does not block queries to avoid false positives.
    """
    query_lower = query.lower()

    for pattern in SUSPICIOUS_PATTERNS:
        if pattern in query_lower:
            logger.warning(f"Suspicious pattern detected in query: '{pattern}'")
            # We log but don't block - users asking about prompt injection are valid
            break

    return query


def route_by_mode(state: AgentState) -> str:
    """Route based on agent mode.

    Args:
        state: Current agent state

    Returns:
        Next node name: "web_search" for online mode, "retriever" for offline
    """
    if state["mode"] == "online":
        return "web_search"
    return "retriever"


def build_graph() -> CompiledStateGraph:
    """Build and compile the agent graph.

    Returns:
        Compiled StateGraph
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_classifier", query_classifier)
    graph.add_node("retriever", retriever)
    graph.add_node("web_search", web_search)
    graph.add_node("answer_generator", answer_generator)

    # Add edges
    # Entry point to query classifier
    graph.add_edge(START, "query_classifier")

    # After classification, route based on mode
    graph.add_conditional_edges(
        "query_classifier", route_by_mode, {"web_search": "web_search", "retriever": "retriever"}
    )

    # After web search, go to retriever (hybrid approach)
    graph.add_edge("web_search", "retriever")

    # After retrieval, generate answer
    graph.add_edge("retriever", "answer_generator")

    # After answer generation, end
    graph.add_edge("answer_generator", END)

    # Compile the graph
    return graph.compile()


# Create a singleton instance of the compiled graph
_graph = None


def get_graph() -> CompiledStateGraph:
    """Get the compiled graph, building it if necessary.

    Returns:
        Compiled StateGraph instance
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str, mode: str = "offline", chat_history: list[dict] | None = None) -> str:
    """Run the agent with a query.

    Args:
        query: The user's question
        mode: "offline" or "online"
        chat_history: Optional list of previous messages

    Returns:
        The agent's response

    Raises:
        ValueError: If query is empty, too long, or mode is invalid
    """
    # Input validation
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed.")

    # Basic prompt injection protection
    query = sanitize_query(query)

    if mode not in ("offline", "online"):
        raise ValueError("Mode must be 'offline' or 'online'")

    # Limit chat history to prevent unbounded growth
    history = chat_history or []
    if len(history) > MAX_CHAT_HISTORY:
        history = history[-MAX_CHAT_HISTORY:]

    graph = get_graph()

    initial_state = {
        "query": query,
        "query_type": None,
        "mode": mode,
        "retrieved_docs": None,
        "web_results": None,
        "context": None,
        "response": None,
        "chat_history": history,
    }

    result = graph.invoke(initial_state)
    return result["response"]
