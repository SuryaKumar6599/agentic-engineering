# Agent Evaluation & Observability Harness

> **Production-grade eval suite for any tool-using LLM agent.**
> Covers task success, tool-call accuracy, cost gating, latency budgets, and nightly regression CI.

---

## Table of Contents
- [What This App Does & Why Evals Matter](#what-this-app-does--why-evals-matter)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Metric Reference](#metric-reference)
- [Adding Your Agent](#adding-your-agent)
- [Golden Dataset Format](#golden-dataset-format)
- [CI Pipeline](#ci-pipeline)
- [Langfuse Observability](#langfuse-observability)
- [promptfoo Prompt Regression](#promptfoo-prompt-regression)
- [Cost & Latency Gates](#cost--latency-gates)
- [HTML Report](#html-report)

---

## What This App Does & Why Evals Matter

Traditional software tests check exact outputs (`assert f(2) == 4`), which breaks down for AI agents because LLMs are non-deterministic, call external tools in multiple steps, and can sound confident while hallucinating or burning excessive tokens. This harness gives you a repeatable, automated way to measure whether an agent is **accurate, reliable, fast, and cost-effective** before and after every code, prompt, or model change.

### How the Evaluations Work Together

- **Task Success (LLM-as-a-Judge via DeepEval & GEval)**
  - **What it checks:** Evaluates the agent's final response for *Answer Relevancy* (did it answer what was asked?), *Faithfulness* (did it stick to retrieved facts without inventing details?), *Task Completion*, and *Correctness* against a golden answer.
  - **How it helps:** Catches subtle quality regressions—such as vague answers, missed instructions, or hallucinations—that simple string matching cannot detect.

- **Tool-Call Accuracy (Deterministic Metrics)**
  - **What it checks:** Inspects the agent's execution trace without needing an LLM judge:
    - *Tool Call Presence:* Did the agent invoke the expected tools (e.g., `calculator`, `search_web`)?
    - *Tool Call Order:* When tasks require sequential steps, did tools run in the right order?
    - *Argument Coverage:* Were required parameters passed and forbidden parameters avoided?
    - *Hallucination Rate:* Did the agent invent or call unnecessary tools?
  - **How it helps:** Verifies the *reasoning and action path* of the agent, not just its final text—critical because an agent can guess a lucky answer while calling the wrong API.

- **RAG & Retrieval Quality (Ragas)**
  - **What it checks:** Evaluates retrieval-augmented agents across *Context Precision* (are relevant documents ranked highest?), *Context Recall* (did retrieval find all needed facts?), *Faithfulness*, and *Answer Relevancy*.
  - **How it helps:** Separates **retrieval failures** (the right document wasn't fetched) from **generation failures** (the LLM ignored or misread the fetched document).

- **Prompt & Model Regression Testing (promptfoo)**
  - **What it checks:** Runs the same test cases across multiple system prompts (e.g., direct vs. chain-of-thought) and multiple models (e.g., `qwen2.5:7b` vs. `qwen3.6:27b`) side-by-side using assertions like `contains`, `is-json`, `llm-rubric`, and latency thresholds.
  - **How it helps:** Lets you safely A/B test prompt edits or swap to smaller/faster local models with objective pass/fail evidence.

- **Cost & Latency Budgets**
  - **What it checks:** Tracks token usage per run (`cost_usd`) and enforces P95/P99 latency limits (`LATENCY_P95_MAX_S`) plus a hard spend cap (`COST_BUDGET_USD`).
  - **How it helps:** Prevents infinite tool-call loops, bloated prompts, or slow model upgrades from silently degrading production UX or blowing up API bills.

- **Observability & Trace Scoring (Langfuse)**
  - **What it checks:** Records every agent run as a structured trace (prompts, tool calls, token counts, latency) and attaches each numeric eval score back to the trace.
  - **How it helps:** When a test fails, you can inspect the exact intermediate step, tool argument, and LLM response that caused the failure.

- **Automated Regression Gates in CI (GitHub Actions)**
  - **What it checks:** Runs fast deterministic unit tests on every commit, and full LLM-judged regression + promptfoo suites on merges and nightly schedules (failing the build if pass rate drops below 80% or budgets are exceeded).
  - **How it helps:** Turns agent quality into a measurable engineering gate rather than manual "vibe checking."

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       eval run / pytest                            │
│                                                                    │
│  GoldenCase  ──►  Agent (your impl)  ──►  AgentTrace              │
│                          │                     │                   │
│                     Langfuse Tracer        Metric Runner           │
│                     (tracing + scores)          │                  │
│                                         ┌───────┴────────┐        │
│                                         │  DeepEval       │        │
│                                         │  Ragas          │        │
│                                         │  Custom Tool    │        │
│                                         │  Accuracy       │        │
│                                         └───────┬────────┘        │
│                                                 ▼                  │
│                                         EvalResult (pass/fail)     │
│                                         + cost_usd + latency_ms    │
└────────────────────────────────────────────────────────────────────┘
         ▼                        ▼                    ▼
   results/latest.jsonl    Langfuse traces      promptfoo comparison
         ▼
   HTML Report / CI JUnit XML / PR comment
```

### Directory Layout

```
.
├── harness/                    # Core library
│   ├── config.py               # Settings (env vars)
│   ├── models.py               # Pydantic models (AgentTrace, GoldenCase, EvalResult)
│   ├── tracer.py               # Langfuse tracing wrapper
│   ├── runner.py               # Async eval runner (concurrency, budgets)
│   ├── dataset.py              # Load/save golden JSONL datasets
│   ├── cost.py                 # Per-token cost table + accumulator
│   ├── cli.py                  # CLI: eval run / report / budget
│   └── metrics/
│       ├── task_success.py     # DeepEval + Ragas LLM-judged metrics
│       └── tool_accuracy.py    # Deterministic tool-call metrics
│
├── agents/
│   └── example_agent/          # Swap in your agent here
│       └── agent.py            # Must expose: async def run(input: str) -> AgentTrace
│
├── evals/
│   ├── fixtures/golden/        # Golden JSONL datasets
│   ├── deepeval/               # DeepEval pytest tests
│   ├── ragas/                  # Ragas RAG tests
│   ├── langfuse/               # Langfuse integration tests
│   └── promptfoo/              # promptfoo.yaml prompt comparison
│
├── tests/
│   └── regression/             # Regression gate: pass-rate, latency, cost
│
├── scripts/
│   └── generate_report.py      # HTML dashboard generator
│
├── .github/workflows/
│   └── eval-ci.yml             # Full CI pipeline
│
├── docker-compose.yml          # Self-hosted Langfuse + eval runner
└── docker/Dockerfile
```

---

## Quick Start

### 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --no-compile -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run unit tests (no API keys needed)

```bash
pytest -m unit -q
```

### 4. Run the full eval suite

```bash
agent-eval run --dataset evals/fixtures/golden/qa.jsonl --agent example_agent
```

### 5. Check cost budget

```bash
agent-eval budget
```

### 6. Generate HTML report

```bash
python3 scripts/generate_report.py
open results/report.html
```

### 7. Run promptfoo prompt comparison

```bash
cd evals/promptfoo
npx promptfoo eval --config promptfoo.yaml
npx promptfoo view
```

### 8. Start self-hosted Langfuse

```bash
docker compose up -d
# Open http://localhost:3000
```

---

## Metric Reference

| Metric | Category | Judge | Threshold |
|--------|----------|-------|-----------|
| `answer_relevancy` | Task success | LLM (DeepEval) | 0.7 |
| `faithfulness` | Task success | LLM (DeepEval) | 0.7 |
| `task_completion` | Task success | LLM (GEval) | 0.7 |
| `correctness` | Task success | LLM (GEval) | 0.8 |
| `ragas_answer_relevancy` | RAG quality | LLM (Ragas) | 0.7 |
| `tool_call_presence` | Tool accuracy | Deterministic | 1.0 |
| `tool_call_order` | Tool accuracy | Deterministic | 1.0 |
| `tool_arg_coverage` | Tool accuracy | Deterministic | 1.0 |
| `tool_hallucination_rate` | Tool accuracy | Deterministic | 1.0 |
| `latency_ok` | Performance | Deterministic | configurable |

All thresholds are configurable via environment variables or per-test overrides.

---

## Adding Your Agent

1. Create `agents/my_agent/agent.py`:

```python
from harness.models import AgentTrace

async def run(task_input: str) -> AgentTrace:
    # ... your agent logic ...
    return AgentTrace(
        trace_id="...",
        task_id="",           # filled in by runner
        input=task_input,
        output="...",
        tool_calls=[...],
        latency_ms=...,
        prompt_tokens=...,
        completion_tokens=...,
        model="gpt-4o",
    )
```

2. Run against it:

```bash
eval run --agent my_agent --dataset evals/fixtures/golden/qa.jsonl
```

---

## Golden Dataset Format

Each line in a `.jsonl` file is a `GoldenCase`:

```jsonc
{
  "task_id": "qa-001",
  "input": "What is 17 * 23?",
  "expected_output": "391",          // optional – for correctness metric
  "expected_tool_calls": [
    {
      "name": "calculator",
      "required_args": ["expression"],   // must be in args
      "forbidden_args": []               // must NOT be in args
    }
  ],
  "expected_tool_order": false,        // true → sequence must match
  "tags": ["math", "tool-use"]
}
```

---

## CI Pipeline

The GitHub Actions workflow (`.github/workflows/eval-ci.yml`) runs:

| Job | Trigger | Gate |
|-----|---------|------|
| Unit tests | Every push/PR | All pass |
| Integration tests | main/dev push + nightly | All pass |
| Regression suite | main/dev push + nightly | Pass rate ≥ 80%, P95 ≤ 15s, cost ≤ $2 |
| promptfoo regression | main/dev push + nightly | All assertions pass |
| PR comment | PRs | Posts markdown summary |

Add these secrets to your GitHub repo:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_HOST`

---

## Langfuse Observability

Every agent run is traced automatically. Each trace contains:
- Full input/output
- Per-generation token usage + cost
- Tool call spans
- Numeric scores for every metric (visible in Langfuse UI → Scores tab)

Self-host with `docker compose up -d` or use [Langfuse Cloud](https://cloud.langfuse.com).

---

## promptfoo Prompt Regression

Compares multiple prompt variants and model providers side-by-side.
Assertions include:
- `contains` / `not-contains`
- `llm-rubric` (LLM-as-judge)
- `is-json` / `javascript` (structural checks)
- `cost` / `latency` (budget enforcement)

---

## Cost & Latency Gates

| Setting | Env var | Default |
|---------|---------|---------|
| Budget per CI run | `COST_BUDGET_USD` | $2.00 |
| P95 latency limit | `LATENCY_P95_MAX_S` | 10s |
| P99 latency limit | `LATENCY_P99_MAX_S` | 20s |
| Concurrency | `EVAL_MAX_CONCURRENCY` | 5 |

The CLI `eval budget` command exits with code 1 if over budget (CI-safe).

---

## HTML Report

```bash
python scripts/generate_report.py --results results/latest.jsonl --out results/report.html
```

Generates an interactive Chart.js dashboard with:
- Pass/fail donut chart
- Average score per metric (bar chart)
- Latency histogram
- Cost per case bar chart
- Full case detail table
