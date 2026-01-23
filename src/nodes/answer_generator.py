"""Answer generation node."""

from loguru import logger

from src.common.prompts import SYSTEM_PROMPT
from src.llm_client import UnifiedLLMClient
from src.llm_client.utils import LLMClientError, ProviderError
from src.state import AgentState


def format_chat_history(history: list) -> str:
    """Format chat history for the prompt.

    Args:
        history: List of message dictionaries with 'role' and 'content'

    Returns:
        Formatted chat history string
    """
    if not history:
        return "No previous conversation."

    formatted = []
    for msg in history[-6:]:  # Keep last 6 messages for context
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def answer_generator(state: AgentState) -> dict:
    """Generate a response based on the retrieved context.

    Args:
        state: Current agent state with query and context

    Returns:
        Updated state with response
    """
    query = state["query"]
    context = state.get("context", "No context available.")
    chat_history = state.get("chat_history", [])

    logger.debug(f"Context length: {len(context)} chars")
    logger.debug(f"Context preview: {context[:500]}...")

    try:
        client = UnifiedLLMClient()

        system_message = SYSTEM_PROMPT.format(
            context=context, chat_history=format_chat_history(chat_history)
        )

        prompt = f"{system_message}\n\nUser Question: {query}"

        response = client.invoke(prompt)

        return {"response": response}

    except ProviderError as e:
        logger.error(f"LLM provider error: {e}")
        return {"response": f"Error: LLM provider unavailable. {e}"}
    except LLMClientError as e:
        logger.error(f"LLM client error: {e}")
        return {"response": f"Error: Unable to generate response. {e}"}
