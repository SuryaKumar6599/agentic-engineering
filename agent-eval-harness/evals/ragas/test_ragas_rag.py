"""
evals/ragas/test_ragas_rag.py
──────────────────────────────
RAG-specific Ragas metrics for agents that retrieve context.

Metrics:
  • answer_relevancy      – is the answer relevant to the question?
  • faithfulness          – is the answer supported by retrieved docs?
  • context_precision     – are retrieved docs ranked by relevance?
  • context_recall        – are all ground-truth facts in the context?

Local model routing
───────────────────
When EVAL_MODEL=local, Ragas is pointed at the local OpenAI-compatible
server via OPENAI_API_KEY / OPENAI_BASE_URL env vars set at import time.

Run:
    pytest evals/ragas/test_ragas_rag.py -m integration
"""

from __future__ import annotations

import os

import pytest

from harness.config import settings

# ── Route Ragas through local server when EVAL_MODEL=local ────────────────────
# Ragas reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment.
if settings.is_local:
    os.environ.setdefault("OPENAI_API_KEY", settings.local_model_api_key)
    os.environ.setdefault("OPENAI_BASE_URL", settings.local_model_base_url)

from ragas import evaluate as ragas_evaluate  # noqa: E402  (must come after env setup)
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from datasets import Dataset


# ── Fixtures (replace with your real RAG golden set) ─────────────────────────

RAG_GOLDEN = [
    {
        "question": "What is the boiling point of water at sea level?",
        "answer": "100°C (212°F) at sea level.",
        "contexts": [
            "Water boils at 100 degrees Celsius (212 °F) at standard atmospheric pressure (1 atm, sea level).",
            "The boiling point of water decreases with altitude.",
        ],
        "ground_truth": "Water boils at 100°C at sea level.",
    },
    {
        "question": "Who wrote 'Pride and Prejudice'?",
        "answer": "Jane Austen wrote 'Pride and Prejudice', published in 1813.",
        "contexts": [
            "Pride and Prejudice is a romantic novel by Jane Austen, published in 1813.",
        ],
        "ground_truth": "Jane Austen",
    },
]


@pytest.mark.integration
@pytest.mark.slow
async def test_ragas_rag_metrics():
    """
    Run Ragas evaluation on the golden RAG dataset.
    All metric scores must exceed 0.5.
    """
    dataset = Dataset.from_list(RAG_GOLDEN)

    # Pass the resolved model name (local or cloud)
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[
            AnswerRelevancy(),
            Faithfulness(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=settings.resolved_model_name,
        embeddings=None,       # uses the same base_url if OPENAI_BASE_URL is set
        raise_exceptions=False,
    )

    for metric_name, score in result.items():
        assert score >= 0.5, f"Ragas metric '{metric_name}' below threshold: {score:.3f}"
        print(f"  {metric_name}: {score:.3f}")
