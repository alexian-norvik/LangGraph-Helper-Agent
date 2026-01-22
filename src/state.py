"""Agent state definition for the LangGraph Helper Agent."""

from typing import Literal, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """State shared across all nodes in the graph.

    Attributes:
        query: The user's original question
        query_type: Classification of the query (langgraph, langchain, code_example, general)
        mode: Current agent mode (offline or online)
        retrieved_docs: Documents retrieved from the vector store
        web_results: Results from web search (online mode only)
        context: Combined context from retrieval and/or web search
        response: The final generated response
        chat_history: Optional list of previous messages for memory
    """

    # Input
    query: str

    # Classification
    query_type: Literal["langgraph", "langchain", "code_example", "general"] | None

    # Mode
    mode: Literal["offline", "online"]

    # Retrieval
    retrieved_docs: list[Document] | None
    web_results: list[str] | None
    context: str | None

    # Output
    response: str | None

    # Memory (optional)
    chat_history: list[dict] | None
