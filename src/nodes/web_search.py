"""Web search node for online mode."""

from ddgs import DDGS
from loguru import logger

from src.config import MAX_SEARCH_RESULTS
from src.state import AgentState


def web_search(state: AgentState) -> dict:
    """Search the web for relevant information.

    Args:
        state: Current agent state with the user's query

    Returns:
        Updated state with web_results
    """
    query = state["query"]
    query_type = state.get("query_type", "general")

    # Enhance query based on classification
    if query_type == "langgraph":
        search_query = f"LangGraph {query}"
    elif query_type == "langchain":
        search_query = f"LangChain {query}"
    else:
        search_query = f"LangGraph LangChain {query}"

    logger.debug(f"Web search query: '{search_query}'")

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    search_query,
                    max_results=MAX_SEARCH_RESULTS,
                    region="wt-wt",  # No region bias
                )
            )

        logger.debug(f"Web search returned {len(results)} results")

        # Format results
        web_results = []
        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")
            href = result.get("href", "")

            if title and body:
                formatted = f"**{title}**\n{body}"
                if href:
                    formatted += f"\nSource: {href}"
                web_results.append(formatted)

        logger.debug(f"Formatted {len(web_results)} web results")
        return {"web_results": web_results}

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"web_results": []}
