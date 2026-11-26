# Agentic RAG Pipeline

Context-aware task detection and workflow segmentation via a true ReAct agent loop backed by FAISS vector retrieval.

## Architecture

```
Query
  │
  ▼
AgenticRAGAgent.run()
  │
  ├── [RAG mode]  pre-fetch context via FAISSRetriever (MMR)
  │   └── is_loaded? No → zero-shot branch with warning
  │
  ▼
create_react_agent  (LangChain ReAct loop)
  ├── Tool: vector_search   → FAISS MMR retrieval, returns JSON
  ├── Tool: classify_task   → TaskDetector (.with_structured_output)
  └── Tool: summarize_context → synthesis
  │   (agent may loop: re-query if context is insufficient)
  │
  ▼
Post-processing (deterministic)
  ├── TaskDetector.detect()       → TaskResult (Pydantic)
  └── WorkflowSegmenter.segment() → WorkflowPlan (DAG validated via networkx)
  │
  ▼
AgentResponse { answer, task, workflow, sources, mode }
```

## Quickstart

### 1. Install

```bash
cd /Users/suryakumar/Projects/agentic-rag-pipeline

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set MOONSHOT_API_KEY
```

Get a free key at [platform.moonshot.cn](https://platform.moonshot.cn).

### 3. Ingest documents

```bash
uv run python main.py --ingest data/sample_docs/
```

### 4. Query

```bash
# One-shot
uv run python main.py --query "What tasks are involved in building a REST API?"

# Interactive REPL
uv run python main.py
```

## Tests

```bash
uv run pytest tests/ -v
```

The test suite runs entirely offline (no API calls, no PyTorch model loading) via mocks.

## Configuration

All settings in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MOONSHOT_API_KEY` | — | Kimi free-tier API key |
| `MOONSHOT_MODEL` | `moonshot-v1-8k` | Model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local HuggingFace model |
| `AGENT_MAX_ITERATIONS` | `6` | ReAct loop cap |
| `RETRIEVAL_K` | `5` | MMR result count |
| `RETRIEVAL_LAMBDA_MULT` | `0.6` | Diversity vs. relevance balance |

## Extending

**Add a task category**: extend `TaskCategory` enum in `schemas/models.py` — the prompt uses it automatically.

**Add a document format**: add an entry to `_LOADERS` in `pipeline/ingestion.py`.

**Swap LLM provider**: change `MOONSHOT_BASE_URL` and `MOONSHOT_MODEL` in `.env` — the code uses the OpenAI-compatible interface throughout (Groq, OpenAI, LM Studio all work).
