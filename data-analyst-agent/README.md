# 🔬 Sandboxed Data-Analyst Agent with Guardrails

A production-grade AI agent that accepts a **CSV or SQLite database** and a **plain-English business question**, then autonomously writes and executes sandboxed SQL and Python, validates every numerical claim against execution evidence, and returns a sourced, reproducible markdown report — **entirely locally, no cloud required**.

```
Input: CSV/SQLite + business question
       ↓
  Schema Profiler  →  LLM Plan  →  SQL Sandbox  →  Python Sandbox
                                         ↓                ↓
                                    Guardrails  ←  Execution Results
                                         ↓
                                 LLM Synthesis  →  Claim Verifier
                                         ↓
                               Validated Report.md
```

> **Tested with Qwen2.5:7b (local Ollama), OpenAI GPT-4o-mini, Anthropic Claude, and Google Gemini.**
> **54/54 offline unit tests pass with zero API key or network access.**

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Guardrail Catalogue](#guardrail-catalogue)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Supported LLM Providers](#supported-llm-providers)
8. [Running Tests](#running-tests)
9. [Failure-Analysis — Real Bugs Caught](#failure-analysis--real-bugs-caught)
10. [Configuration Reference](#configuration-reference)
11. [Extending the Agent](#extending-the-agent)

---

## Architecture

### Two-Phase Agent Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                        AnalystAgent                             │
│                                                                 │
│  for attempt in 1 .. max_retries + 1:                          │
│                                                                 │
│    ┌──────────────┐    JSON plan     ┌──────────────────────┐   │
│    │  LLM PLAN    │ ───────────────► │   SQLSandbox         │   │
│    │  (system     │                  │   • keyword blocker  │   │
│    │   prompt +   │                  │   • set_authorizer   │   │
│    │   schema     │                  │   • progress timeout │   │
│    │   profile)   │                  └──────────┬───────────┘   │
│    └──────────────┘                             │ SQLResult     │
│           ▲                                     ▼               │
│           │ retry feedback              ┌──────────────────────┐ │
│           │ (errors + column hints)     │   PythonSandbox      │ │
│           │                            │   • AST validator    │ │
│    ┌──────┴───────┐                    │   • python -I        │ │
│    │  Guardrails  │ ◄─────────────── │   • resource limits  │ │
│    │  + Claim     │   findings        │   • DATA injection   │ │
│    │  Verifier    │                    └──────────────────────┘ │
│    └──────┬───────┘                                             │
│           │ no errors                                           │
│           ▼                                                     │
│    ┌──────────────┐    markdown      ┌──────────────────────┐   │
│    │  LLM         │ ───────────────► │  Claim-to-Evidence   │   │
│    │  SYNTHESIS   │                  │  Verifier            │   │
│    └──────────────┘                  └──────────┬───────────┘   │
│                                                 │               │
│                                        PASS ────┘               │
│                                        FAIL → retry with feedback│
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
CSV / SQLite
    │
    ▼
DataSourceManager.profile()
    │  → ColumnProfile (type, nulls, distinct count, min/max, samples)
    │  → TableProfile  (row count, PK candidates)
    │  → DatabaseProfile.to_prompt_context()  ← injected into LLM prompts
    │
    ▼
SQLite in-memory DB (read-only URI + PRAGMA query_only=ON)
    │
    ├── SQL queries executed via SQLSandbox (read-only authorizer)
    │       → SQLResult.rows  (capped at max_rows, truncation flagged)
    │
    └── Python scripts executed via PythonSandbox (isolated subprocess)
            DATA["q1"] = SQLResult.rows  ← pre-injected
            record_metric("name", value) ← surfaces numbers for verification
            → PythonResult.metrics
```

---

## Project Structure

```
.
├── pyproject.toml                    # PEP 517 build config, optional deps
├── README.md
├── report.md                         # Latest generated report (gitignore this)
│
├── sample_data/
│   ├── orders.csv                    # 30-row sales orders (revenue, category, region, date)
│   └── customers.csv                 # 20-row customer master (tier, churn_risk_score)
│
├── src/analyst_agent/
│   ├── __init__.py                   # Public API surface
│   ├── models.py                     # Dataclasses: ColumnProfile → AnalysisReport
│   ├── profiler.py                   # CSV → SQLite ingestion + schema profiler
│   ├── sql_sandbox.py                # Read-only SQL executor with dual-layer security
│   ├── python_sandbox.py             # AST-validated isolated Python subprocess
│   ├── guardrails.py                 # 12 guardrail codes + claim-to-evidence verifier
│   ├── agent.py                      # Orchestrator: plan → execute → guard → synthesise → retry
│   ├── llm.py                        # LLM adapters: Ollama, OpenAI, Anthropic, Gemini
│   └── cli.py                        # `analyst-agent` CLI entrypoint
│
└── tests/
    ├── test_agent.py                 # pytest-compatible test file
    └── run_tests.py                  # Stdlib-only runner (no pytest needed), 54 tests
```

---

## How It Works

### 1. Schema Profiling (`profiler.py`)

The `DataSourceManager` ingests CSV files or directories into an **in-memory SQLite database**, then profiles every column:

- **Type inference** — scans up to 1,000 sample values, classifies as `INTEGER / REAL / TEXT`
- **Null analysis** — exact null count, null rate; triggers `HIGH_NULL_RATE` warning at ≥20%
- **Distinct count + range** — used for join fanout detection and evidence matching
- **Sample values** — 5 representative values injected into the LLM context
- **Primary key candidates** — columns with 0 nulls and 100% distinct values

The profile is serialised into a structured prompt context (not raw SQL DDL) so LLMs cannot infer ambiguous schema from implicit CREATE TABLE syntax.

### 2. SQL Sandbox (`sql_sandbox.py`)

Two independent security layers:

| Layer | Mechanism | Catches |
|-------|-----------|---------|
| **Static** | Regex keyword blocklist on comment/string-stripped SQL | `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `ATTACH`, `TRIGGER`, multi-statement |
| **Runtime** | SQLite `set_authorizer` callback — returns `SQLITE_DENY` for any action other than `SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION` | Edge cases that evade keyword matching |

Additional protections:
- `progress_handler` aborts queries exceeding `timeout_seconds` (default 5s)
- Rows capped at `max_rows` (default 500); truncation flagged as a guardrail warning
- Connection opened with `?mode=ro` URI + `PRAGMA query_only=ON`
- Authorizer and progress handler always unset in `finally:` to prevent state leakage

### 3. Python Sandbox (`python_sandbox.py`)

Three-layer isolation:

| Layer | Mechanism |
|-------|-----------|
| **AST pre-check** | Walks the AST before subprocess launch; rejects disallowed `import`, forbidden builtins (`eval`, `exec`, `open`, `__import__`), and dangerous attribute access (`.system`, `.__globals__`, `.__code__`) |
| **Subprocess isolation** | `python3 -I` (isolated mode: no user site-packages, ignores `PYTHONPATH`) with stripped `PATH` env |
| **Resource limits** | `resource.setrlimit(RLIMIT_CPU, N)` inside the subprocess caps CPU seconds |

Data flow into scripts:
```python
# The agent injects this automatically — the LLM script just uses DATA
DATA = {"q1": [{"category": "Electronics", "total": 599.98}, ...]}

# Scripts surface numbers via record_metric() — these feed the claim verifier
record_metric("electronics_revenue", DATA["q1"][0]["total"])
```

Allowed imports: `math`, `statistics`, `json`, `csv`, `datetime`, `collections`, `itertools`, `functools`, `decimal`, `re`, `random`, `numpy`, `pandas`, `scipy`

### 4. Guardrails (`guardrails.py`)

Run after every execution attempt, before synthesis:

#### Execution Guardrails

| Code | Trigger | Severity |
|------|---------|----------|
| `SCHEMA_HALLUCINATION` | SQL error contains "no such table/column" | ERROR |
| `SQL_EXECUTION_ERROR` | Any other SQL runtime error | ERROR |
| `NO_VALID_SQL_EVIDENCE` | Every SQL query failed | ERROR |
| `EMPTY_RESULT` | Query returns 0 rows | WARNING |
| `TRUNCATED_RESULT` | Row cap hit | WARNING |
| `JOIN_FANOUT_EXPLOSION` | JOIN result > largest input table row count | ERROR |
| `NON_FINITE_VALUE` | NaN or Inf in result column | ERROR |
| `HIGH_NULL_RATE` | Column has ≥20% missing values | WARNING |
| `PYTHON_EXECUTION_ERROR` | Subprocess non-zero exit | ERROR |
| `NULL_OR_NAN_METRIC` | `record_metric()` received None/NaN | WARNING |
| `SMALL_SAMPLE_SIZE` | Sample-size metric < 30 | WARNING |

#### Claim-to-Evidence Verifier

After synthesis, every number in the markdown narrative is extracted via regex and cross-checked against an **evidence set** built from SQL rows, Python metrics, and the DatabaseProfile:

- Tolerance: `rel_tol=1.5%` or `abs_tol=0.05` (handles rounding)
- Percentage ↔ decimal bridging: `0.25` in metrics matches `25%` in prose
- Year-label suppression: integers 1990–2035 without `%`/`$` are skipped
- Any unmatched number → `UNGROUNDED_NUMERICAL_CLAIM` ERROR

### 5. Self-Correction Loop (`agent.py`)

```
attempt 1:
  plan → execute → guardrails → [errors?] → synthesise → claim-verify → [errors?]
                      └── ERROR ──────────────────────────────────────────┘
                           ↓
                    build retry feedback:
                    - guardrail error messages
                    - actual SQL column names + sample row  ← key for local LLMs
                           ↓
attempt 2: plan (with feedback) → ...
```

The retry feedback injects the **actual produced column names** so the LLM can fix Python `KeyError`-style bugs without hallucinating column names it never saw.

---

## Guardrail Catalogue

```
SCHEMA_HALLUCINATION      → SQL referenced nonexistent table/column
SQL_EXECUTION_ERROR       → SQL runtime error (non-hallucination)
NO_VALID_SQL_EVIDENCE     → All queries failed; synthesis ungrounded
EMPTY_RESULT              → Query returned 0 rows
TRUNCATED_RESULT          → Row cap reached; aggregation needed
JOIN_FANOUT_EXPLOSION     → JOIN produced more rows than largest input table
NON_FINITE_VALUE          → NaN/Inf in result column
HIGH_NULL_RATE            → Column ≥ 20% missing; affects aggregation denominators
PYTHON_EXECUTION_ERROR    → Subprocess non-zero exit
NULL_OR_NAN_METRIC        → record_metric() value is None/NaN
SMALL_SAMPLE_SIZE         → n < 30; statistical interpretation unreliable
UNGROUNDED_NUMERICAL_CLAIM → Narrative number not in SQL/Python evidence set
```

---

## Installation

**Requirements:** Python 3.11+ (uses `sqlite3`, `ast`, `subprocess`, `json` — all stdlib)

```bash
git clone <repo-url>
cd "Deep research or data-analyst agent with guardrails"

# No pip install needed for local Ollama usage or running tests.
# Optional: install for CLI entrypoint
pip install -e .

# Optional: data-science extras (pandas, numpy)
pip install -e ".[ds]"

# Optional: cloud LLM clients
pip install -e ".[llm]"
```

---

## Usage

### Option A — Local Ollama (recommended, free, private)

```bash
# 1. Make sure Ollama is running
ollama serve          # or it may already be running as a background service
ollama pull qwen2.5:7b

# 2. Run the agent
PYTHONPATH=src python3 src/analyst_agent/cli.py \
  --source sample_data/ \
  --question "Which product category drives the most revenue?" \
  --provider ollama \
  --model qwen2.5:7b \
  --out report.md

open report.md
```

**Other available local models:**
```bash
--model qwen3.6:27b     # Best quality (27B, needs ~20GB RAM)
--model qwen2.5vl:7b    # Vision-capable
--model llama3.2:3b     # Fastest (3B)
--model mistral:latest  # Good alternative (7B)
```

### Option B — Cloud LLM

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
PYTHONPATH=src python3 src/analyst_agent/cli.py \
  --source sample_data/ \
  --question "Which category has the highest churn risk customers?" \
  --provider openai \
  --model gpt-4o-mini \
  --out report.md

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=src python3 src/analyst_agent/cli.py \
  --source my_data.db \
  --question "What is the month-over-month revenue trend?" \
  --provider anthropic

# Google Gemini
export GEMINI_API_KEY=...
PYTHONPATH=src python3 src/analyst_agent/cli.py \
  --source sales.csv \
  --question "Top 3 regions by revenue?" \
  --provider gemini \
  --model gemini-2.5-flash
```

### Option C — Python API

```python
from analyst_agent import AnalystAgent
from analyst_agent.llm import OllamaClient, OpenAIClient

# Local Ollama
agent = AnalystAgent(
    llm=OllamaClient(model="qwen2.5:7b"),
    max_retries=2,
    max_sql_rows=500,
    sql_timeout=5.0,
    python_timeout=10.0,
)

report = agent.analyze(
    source_path="sample_data/",           # CSV dir, single CSV, or .sqlite file
    question="Which region has the highest revenue?",
)

print(report.to_markdown())
print(f"Passed: {report.validation_passed}, Attempts: {report.attempts}")
print(f"Findings: {[(f.code, f.severity) for f in report.findings]}")
```

### CLI Reference

```
usage: analyst-agent [-h] --source SOURCE --question QUESTION
                     [--provider {openai,anthropic,gemini,ollama}]
                     [--model MODEL] [--max-retries MAX_RETRIES]
                     [--out OUT]

Arguments:
  --source      Path to CSV file, directory of CSV files, or .sqlite/.db file
  --question    Business question in plain English
  --provider    LLM backend (default: auto-detect from env vars, then Ollama)
  --model       Model name override (e.g. qwen2.5:7b, gpt-4o-mini)
  --max-retries Number of self-correction retries (default: 2)
  --out         Output markdown file path (default: print to stdout)
```

---

## Supported LLM Providers

| Provider | Class | Auth | Notes |
|----------|-------|------|-------|
| **Ollama** | `OllamaClient` | None | Local, private, free. Requires `ollama serve`. |
| **OpenAI** | `OpenAIClient` | `OPENAI_API_KEY` | Default model: `gpt-4o-mini` |
| **Anthropic** | `AnthropicClient` | `ANTHROPIC_API_KEY` | Default model: `claude-3-5-sonnet-latest` |
| **Google Gemini** | `GeminiClient` | `GEMINI_API_KEY` | Default model: `gemini-2.5-flash` |
| **Scripted** | `ScriptedLLMClient` | None | Deterministic test double for offline testing |

**Using a custom LLM:**
```python
from analyst_agent.llm import LLMClient

class MyLLMClient:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Call any LLM API here
        return response_text

agent = AnalystAgent(llm=MyLLMClient())
```

---

## Running Tests

All 54 tests run **fully offline** — no API key, no network, no pip installs beyond stdlib.

```bash
cd "Deep research or data-analyst agent with guardrails"

# Option 1: stdlib runner (always works)
python3 tests/run_tests.py

# Option 2: pytest (if installed)
PYTHONPATH=src python3 -m pytest tests/test_agent.py -v
```

**Test coverage:**

| Suite | Tests | What's covered |
|-------|-------|---------------|
| Profiler | 12 | CSV ingestion, type inference, null/distinct counts, error cases |
| SQL Sandbox | 13 | Keyword validation, authorizer, SUM/GROUP BY, truncation, timeout |
| Python Sandbox | 10 | AST validation, metric injection, subprocess isolation, timeout |
| Guardrails | 8 | All ERROR/WARNING codes, clean-run baseline |
| Claim Verifier | 4 | Grounded/ungrounded numbers, year suppression, pct↔decimal bridging |
| JSON Extraction | 2 | Plain JSON, fenced code block stripping |
| Agent E2E | 5 | Happy path, hallucination retry, ungrounded claims, forbidden Python |

---

## Failure-Analysis — Real Bugs Caught

These are **real failures that occurred during development**, caught by the guardrail system:

### F-1 — Schema Hallucination: Invented `products` table
**What happened:** Qwen2.5:7b generated `JOIN products p ON o.product_id = p.product_id` even though no `products` table exists in the data.
**Guardrail:** `SCHEMA_HALLUCINATION` → `SQLiteError: no such table: products`
**Fix:** Stricter PLAN prompt with `NEVER JOIN to a table unless it appears in the DatabaseProfile`

### F-2 — Python KeyError: Wrong column name in DATA access
**What happened:** After getting the SQL right, the model referenced `DATA['q1'][0]['top_category_2024']` but the actual column was named `category`.
**Guardrail:** `PYTHON_EXECUTION_ERROR` → `KeyError: 'top_category_2024'`
**Fix:** Retry feedback now injects the **actual SQL column names + sample row** so the model corrects itself.

### F-3 — Disallowed `import sqlite3` in Python script
**What happened:** The model tried to re-open the database inside the Python script using `import sqlite3; conn = sqlite3.connect(...)`.
**Guardrail:** `PYTHON_EXECUTION_ERROR` → `PythonSandboxError: Disallowed import: 'sqlite3'`
**Fix:** Explicit rule added to PLAN prompt: `Do NOT call sqlite3.connect()`. DATA is pre-injected.

### F-4 — Test: `assert_equal()` called with 3 args
**What happened:** Custom test helper defined with 2 positional args but called with 3 (pytest-style).
**Fix:** Changed to `assert_true(expr, msg)`.

### F-5 — SQLite authorizer not unset in exception path
**What happened:** `set_authorizer(None)` only called on happy path; exception path left authorizer active on subsequent connections.
**Fix:** Moved to `finally:` block alongside `set_progress_handler(None, 0)` and `conn.close()`.

### F-6 — Calendar year `2024` flagged as hallucinated number
**What happened:** The claim verifier regex matched `2024` in prose like _"In 2024, revenue grew."_ — not in any SQL result set → false positive `UNGROUNDED_NUMERICAL_CLAIM`.
**Fix:** Year-label suppression: skip integers 1990–2035 without `%` or `$` prefix.

---

## Configuration Reference

```python
AnalystAgent(
    llm=OllamaClient("qwen2.5:7b"),  # Required: any LLMClient
    max_retries=2,          # Self-correction retry limit (default: 2)
    max_sql_rows=500,       # Row cap per query (default: 500)
    sql_timeout=5.0,        # SQLite query timeout in seconds (default: 5)
    python_timeout=10.0,    # Python subprocess timeout in seconds (default: 10)
)

OllamaClient(
    model="qwen2.5:7b",             # Model name from `ollama list`
    base_url="http://localhost:11434",  # Ollama server URL
    timeout=120,                     # HTTP request timeout in seconds
)
```

---

## Extending the Agent

### Add a new guardrail
In `guardrails.py`, add a new check to `run_execution_guardrails()`:
```python
findings.append(GuardrailFinding(
    code="MY_CUSTOM_CHECK",
    severity=Severity.WARNING,
    artifact_id=sr.query_id,
    message="Description of what went wrong.",
    remediation="How to fix it.",
))
```

### Allow additional Python imports
In `python_sandbox.py`, add to `ALLOWED_MODULES`:
```python
ALLOWED_MODULES = frozenset({
    ...,
    "scipy",   # e.g. for statistical tests
    "sklearn",  # e.g. for ML metrics
})
```

### Use your own data
```bash
# Single CSV
PYTHONPATH=src python3 src/analyst_agent/cli.py --source my_sales.csv ...

# Directory of CSVs (all loaded as separate tables)
PYTHONPATH=src python3 src/analyst_agent/cli.py --source ./data_dir/ ...

# Existing SQLite database
PYTHONPATH=src python3 src/analyst_agent/cli.py --source my_database.db ...
```
