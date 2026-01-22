"""StateGraph definition for the LangGraph Helper Agent."""

from langgraph.graph import END, START, StateGraph

from src.nodes.answer_generator import answer_generator
from src.nodes.query_classifier import query_classifier
from src.nodes.retriever import retriever
from src.nodes.web_search import web_search
from src.state import AgentState


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


def build_graph() -> StateGraph:
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


def get_graph():
    """Get the compiled graph, building it if necessary."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str, mode: str = "offline", chat_history: list = None) -> str:
    """Run the agent with a query.

    Args:
        query: The user's question
        mode: "offline" or "online"
        chat_history: Optional list of previous messages

    Returns:
        The agent's response
    """
    graph = get_graph()

    initial_state = {
        "query": query,
        "query_type": None,
        "mode": mode,
        "retrieved_docs": None,
        "web_results": None,
        "context": None,
        "response": None,
        "chat_history": chat_history or [],
    }

    result = graph.invoke(initial_state)
    return result["response"]
