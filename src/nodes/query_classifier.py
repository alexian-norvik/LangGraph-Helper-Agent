"""Query classification node."""

from src.common.constants import DEFAULT_QUERY_TYPE, VALID_QUERY_TYPES
from src.common.prompts import CLASSIFICATION_PROMPT
from src.llm_client import UnifiedLLMClient
from src.state import AgentState


def query_classifier(state: AgentState) -> dict:
    """Classify the user's query to determine the best retrieval strategy.

    Args:
        state: Current agent state with the user's query

    Returns:
        Updated state with query_type
    """
    query = state["query"]

    client = UnifiedLLMClient()

    response = client.invoke(CLASSIFICATION_PROMPT.format(query=query))
    query_type = response.strip().lower()

    # Validate the classification
    if query_type not in VALID_QUERY_TYPES:
        # Default to langgraph if classification is unclear
        query_type = DEFAULT_QUERY_TYPE

    return {"query_type": query_type}
