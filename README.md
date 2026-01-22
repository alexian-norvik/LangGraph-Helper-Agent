# LangGraph Helper Agent

An AI assistant that helps developers with LangGraph and LangChain questions, supporting both offline (local RAG) and online (web search + RAG) modes.

## Features

- **Query Classification**: Automatically classifies questions as LangGraph, LangChain, code examples, or general
- **Offline Mode**: Uses locally downloaded documentation with FAISS vector search
- **Online Mode**: Combines DuckDuckGo web search with local RAG for up-to-date information
- **Interactive CLI**: Chat-based interface with mode switching
- **Conversation Memory**: Optional multi-turn conversation support

## Architecture

### Graph Design

The agent uses a LangGraph StateGraph with conditional routing:

```
┌─────────────────┐
│     START       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ query_classifier│  ← Classify question type (langgraph/langchain/code_example/general)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  mode_router    │  ← Route based on offline/online mode
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│retriever│ │web_search│  ← Online mode: search first
└────┬───┘ └────┬─────┘
     │          │
     │          ▼
     │     ┌────────┐
     │     │retriever│  ← Hybrid: web + local docs
     │     └────┬───┘
     │          │
     └────┬─────┘
          │
          ▼
┌─────────────────┐
│answer_generator │  ← Generate response with LLM (Ollama/OpenRouter)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      END        │
└─────────────────┘
```

### State Management

The agent uses a TypedDict-based state (`AgentState`) that flows through all nodes:
- `query`: User's original question
- `query_type`: Classification result
- `mode`: Current operating mode (offline/online)
- `retrieved_docs`: Documents from vector store
- `web_results`: Results from web search (online mode)
- `context`: Combined context for answer generation
- `response`: Final generated response
- `chat_history`: Optional conversation history

### Node Structure

| Node | Purpose |
|------|---------|
| `query_classifier` | Uses LLM to classify the question type for better retrieval |
| `web_search` | DuckDuckGo search for real-time information (online mode only) |
| `retriever` | FAISS similarity search over local documentation |
| `answer_generator` | Generates final response using LLM with retrieved context |

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) for local LLM and embeddings (fully offline option)
- Or [OpenRouter](https://openrouter.ai) API key for cloud LLMs

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LangGraph-Helper-Agent.git
   cd LangGraph-Helper-Agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install and set up Ollama (for LLM and embeddings):
   ```bash
   # Option 1: Download from https://ollama.ai (recommended)
   # Option 2: Install via Homebrew (macOS)
   brew install ollama

   # Start Ollama service
   ollama serve  # Or just open the Ollama app

   # Pull the LLM and embedding models
   ollama pull llama3.2           # For LLM
   ollama pull nomic-embed-text   # For embeddings
   ```

4. Download documentation and create the vector store:
   ```bash
   python scripts/prepare_data.py
   ```

## Operating Modes

### Offline Mode

**How it works:**
- Uses pre-downloaded llms.txt documentation files stored locally
- Performs FAISS similarity search to find relevant documentation chunks
- No internet required during query time (LLM API calls still needed)

**Data sources:**
- LangGraph: `https://langchain-ai.github.io/langgraph/llms.txt`
- LangGraph Full: `https://langchain-ai.github.io/langgraph/llms-full.txt`
- LangChain: `https://docs.langchain.com/llms.txt`
- LangChain Full: `https://docs.langchain.com/llms-full.txt`

**Best for:** Stable, well-documented features; working in low-connectivity environments

### Online Mode

**How it works:**
- First performs a DuckDuckGo web search for real-time information
- Then combines web results with local documentation retrieval
- Provides hybrid context to the answer generator

**Services used:**
- **DuckDuckGo Search**: Free, no API key required, privacy-focused

**Best for:** Latest features, recent updates, edge cases not in official docs

### Mode Switching

```bash
# Via CLI flag
python main.py --mode offline "How do I use checkpointers?"
python main.py --mode online "What are the latest LangGraph features?"

# Via environment variable
export AGENT_MODE=online
python main.py "Your question here"

# In interactive mode, type 'mode' to switch
python main.py
> mode
Switched from offline to online mode
```

## Data Freshness Strategy

### Offline Mode

**Data preparation:**
- Documentation is downloaded via `scripts/prepare_data.py`
- Text is chunked using RecursiveCharacterTextSplitter (2000 chars, 200 overlap)
- Chunks are embedded using Ollama (nomic-embed-text) - no rate limits
- FAISS index is created and stored locally

**Updating data:**
```bash
# Re-download docs and rebuild vector store
python scripts/prepare_data.py --force
```

**Recommendation:** Run the update script periodically (e.g., weekly) to keep documentation current.

### Online Mode

**Real-time updates:**
- DuckDuckGo searches provide current information without manual updates
- Web results are combined with local docs for comprehensive answers

**Why DuckDuckGo:**
- Completely free with no API key required
- No rate limits for reasonable usage
- Privacy-focused (no tracking)
- Returns relevant results for technical queries

## Usage

### Interactive Mode

```bash
python main.py
```

Commands in interactive mode:
- `mode` - Switch between offline/online mode
- `quit` or `exit` - Exit the program
- `help` - Show help message

### Single Query

```bash
# Offline mode (default)
python main.py "How do I create a StateGraph?"

# Online mode
python main.py --mode online "What's new in LangGraph?"

# Specify mode explicitly
python main.py --mode offline "How do I add persistence?"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key (only if using OpenRouter) | - |
| `AGENT_MODE` | Default mode: `offline` or `online` | `offline` |
| `ENABLE_MEMORY` | Enable conversation memory | `false` |

### LLM Configuration

The LLM provider is configured in `config.yaml`:

```yaml
llm:
  platform: ollama  # or 'openrouter'
  model:
    name: "llama3.2"  # or any OpenRouter model
  parameters:
    temperature: 0.3
    max_tokens: 2000
```

To switch providers, simply edit `config.yaml` - no code changes needed.

## Project Structure

```
LangGraph-Helper-Agent/
├── main.py                      # CLI entry point
├── config.yaml                  # LLM provider configuration
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── README.md                    # This file
├── data/
│   ├── langgraph-llms.txt       # Downloaded LangGraph docs
│   ├── langgraph-llms-full.txt  # Full LangGraph docs
│   ├── langchain-llms.txt       # Downloaded LangChain docs
│   ├── langchain-llms-full.txt  # Full LangChain docs
│   └── vectorstore/             # FAISS index
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── state.py                 # AgentState TypedDict
│   ├── graph.py                 # StateGraph definition
│   ├── llm_client/              # Unified LLM client (vendored)
│   │   ├── client.py            # UnifiedLLMClient
│   │   ├── schemas.py           # Pydantic models
│   │   └── utils.py             # Utilities and exceptions
│   ├── general_utils/           # General utilities
│   │   └── config_loader.py     # YAML config loading
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── query_classifier.py  # Query understanding
│   │   ├── retriever.py         # RAG retrieval
│   │   ├── web_search.py        # DuckDuckGo search
│   │   └── answer_generator.py  # LLM response generation
│   └── data_prep/
│       ├── __init__.py
│       ├── downloader.py        # Download llms.txt files
│       ├── chunker.py           # Text splitting
│       └── vectorstore.py       # FAISS management
└── scripts/
    └── prepare_data.py          # One-time data setup
```

## Example Questions

- "How do I add persistence to a LangGraph agent?"
- "What's the difference between StateGraph and MessageGraph?"
- "Show me how to implement human-in-the-loop with LangGraph"
- "How do I handle errors and retries in LangGraph nodes?"
- "What are best practices for state management in LangGraph?"

## Technical Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **LLM** | [Ollama](https://ollama.ai) / [OpenRouter](https://openrouter.ai) | Local or cloud, configurable |
| **Embeddings** | [Ollama](https://ollama.ai) (nomic-embed-text) | Local, no rate limits, 8K context |
| **Vector Store** | FAISS | Lightweight, no server needed, fast |
| **Web Search** | DuckDuckGo | Free, no API key required |
| **Framework** | LangGraph + LangChain | The tools we're helping developers with |

## Dependencies

Key package versions (see `requirements.txt` for full list):

```
langgraph>=1.0.0
langchain>=1.0.0
langchain-core>=1.0.0
langchain-community>=0.3.0
langchain-ollama>=0.2.0
pydantic>=2.0.0
pyyaml>=6.0.0
faiss-cpu>=1.7.4
duckduckgo-search>=6.0.0
```

## API Keys & External Services

| Service | Purpose | How to Get | Cost |
|---------|---------|------------|------|
| Ollama | LLM + Embeddings | [ollama.ai](https://ollama.ai) | Free (runs locally) |
| OpenRouter | LLM (alternative) | [openrouter.ai/keys](https://openrouter.ai/keys) | Free tier + paid |
| DuckDuckGo | Web Search | No key needed | Free |

## License

MIT
