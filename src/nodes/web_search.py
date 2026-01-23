"""Web search node for online mode."""

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from loguru import logger
from requests.exceptions import RequestException

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

    except RatelimitException as e:
        logger.warning(f"DuckDuckGo rate limit exceeded: {e}")
        return {"web_results": []}
    except TimeoutException as e:
        logger.warning(f"DuckDuckGo search timeout: {e}")
        return {"web_results": []}
    except DDGSException as e:
        logger.warning(f"DuckDuckGo search error: {e}")
        return {"web_results": []}
    except RequestException as e:
        logger.warning(f"Network error during web search: {e}")
        return {"web_results": []}
    except (TimeoutError, ConnectionError) as e:
        logger.warning(f"Connection error during web search: {e}")
        return {"web_results": []}
