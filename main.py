#!/usr/bin/env python3
"""CLI entry point for the LangGraph Helper Agent."""

import argparse
import sys
import warnings

from loguru import logger

# Suppress noisy warnings from dependencies (FAISS SWIG, Ollama sockets)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*SwigPy.*")
warnings.filterwarnings("ignore", message=".*swigvarlink.*")


def configure_logging(level: str = "WARNING"):
    """Configure loguru logging level.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )


def get_user_choice(prompt: str, options: list[str], default: str = None) -> str:
    """Get a choice from the user.

    Args:
        prompt: The prompt to display
        options: List of valid options
        default: Default option if user presses enter

    Returns:
        The selected option
    """
    options_str = "/".join(options)
    default_str = f" [{default}]" if default else ""

    while True:
        try:
            choice = input(f"{prompt} ({options_str}){default_str}: ").strip().lower()

            if not choice and default:
                return default

            if choice in [o.lower() for o in options]:
                return choice

            print(f"  Invalid choice. Please enter one of: {options_str}")
        except (EOFError, KeyboardInterrupt):
            print("\n")
            sys.exit(0)


def setup_session() -> dict:
    """Interactive setup for the session.

    Returns:
        Dictionary with session configuration
    """
    print("\n" + "=" * 60)
    print("  LangGraph Helper Agent - Setup")
    print("=" * 60)

    # Mode selection
    mode = get_user_choice("\n1. Select mode", ["offline", "online"], default="offline")

    # Memory selection
    memory_choice = get_user_choice("2. Enable conversation memory", ["yes", "no"], default="no")
    enable_memory = memory_choice == "yes"

    # Log level selection
    log_level = get_user_choice(
        "3. Log level", ["quiet", "normal", "verbose", "debug"], default="quiet"
    )

    level_map = {"quiet": "ERROR", "normal": "WARNING", "verbose": "INFO", "debug": "DEBUG"}

    print("\n" + "-" * 60)
    print(f"  Mode: {mode} | Memory: {'on' if enable_memory else 'off'} | Logs: {log_level}")
    print("-" * 60)

    return {"mode": mode, "enable_memory": enable_memory, "log_level": level_map[log_level]}


def run_interactive(mode: str, enable_memory: bool):
    """Run the agent in interactive mode.

    Args:
        mode: Agent mode (offline/online)
        enable_memory: Whether to enable conversation memory
    """
    from src.config import get_mode, set_mode
    from src.graph import run_agent

    set_mode(mode)
    logger.info(f"Starting interactive mode: {mode}, memory={'on' if enable_memory else 'off'}")

    print("\nReady! Type your question or 'help' for commands.\n")

    chat_history = [] if enable_memory else None

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if query.lower() == "help":
            print(f"\n  Mode: {get_mode()} | Memory: {'on' if enable_memory else 'off'}")
            print("  Commands:")
            print("    mode     - Switch offline/online")
            print("    status   - Show current settings")
            print("    clear    - Clear chat history")
            print("    quit     - Exit")
            print()
            continue

        if query.lower() == "status":
            print(f"\n  Mode: {get_mode()}")
            print(f"  Memory: {'on' if enable_memory else 'off'}")
            if chat_history:
                print(f"  History: {len(chat_history)} messages")
            print()
            continue

        if query.lower() == "mode":
            current = get_mode()
            new_mode = "online" if current == "offline" else "offline"
            set_mode(new_mode)
            logger.info(f"Mode switched: {current} -> {new_mode}")
            print(f"\n  Switched to {new_mode} mode\n")
            continue

        if query.lower() == "clear":
            if chat_history is not None:
                chat_history.clear()
                print("\n  Chat history cleared\n")
            else:
                print("\n  Memory is disabled\n")
            continue

        try:
            logger.info(f"Processing: {query[:50]}...")
            response = run_agent(query, mode=get_mode(), chat_history=chat_history)
            print(f"\nAssistant: {response}\n")

            if chat_history is not None:
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}\n")


def run_single_query(query: str, mode: str):
    """Run a single query and print the response."""
    from src.config import set_mode
    from src.graph import run_agent

    set_mode(mode)
    logger.info(f"Single query - mode: {mode}")

    try:
        response = run_agent(query, mode=mode)
        print(response)
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Helper Agent - AI assistant for LangGraph and LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive setup + chat
  python main.py "How do I create a StateGraph?"   # Quick query (offline)
  python main.py -m online "Latest features"       # Quick query (online)

First-time setup:
  python scripts/prepare_data.py   # Download docs and create vector store
        """,
    )

    parser.add_argument(
        "query", nargs="?", help="Question to ask (if omitted, starts interactive mode)"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["offline", "online"],
        default="offline",
        help="Agent mode for quick queries (default: offline)",
    )

    args = parser.parse_args()

    # Quick query mode - no setup, just answer
    if args.query:
        configure_logging("WARNING")

        from src.data_prep.vectorstore import vectorstore_exists

        if not vectorstore_exists():
            logger.warning("Vector store not found")

        run_single_query(args.query, args.mode)
    else:
        # Interactive mode - run setup first
        config = setup_session()
        configure_logging(config["log_level"])

        from src.data_prep.vectorstore import vectorstore_exists

        if not vectorstore_exists():
            logger.warning("Vector store not found. Run 'python scripts/prepare_data.py' first.")

        run_interactive(config["mode"], config["enable_memory"])


if __name__ == "__main__":
    main()
