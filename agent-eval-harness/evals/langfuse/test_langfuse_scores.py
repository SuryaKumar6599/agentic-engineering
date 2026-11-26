"""
evals/langfuse/test_langfuse_scores.py
──────────────────────────────────────
Integration test: run eval suite end-to-end and verify that
Langfuse traces were created with the expected scores.

Requires LANGFUSE_* env vars to be set.
Skip automatically when Langfuse is not configured.
"""

from __future__ import annotations

import os

import pytest

from harness.config import settings
from harness.dataset import load_golden_cases
from harness.metrics.task_success import answer_relevancy, task_completion_geval
from harness.runner import run_eval_suite


SKIP_NO_LANGFUSE = pytest.mark.skipif(
    not settings.langfuse_secret_key,
    reason="LANGFUSE_SECRET_KEY not set",
)


@SKIP_NO_LANGFUSE
@pytest.mark.integration
@pytest.mark.slow
async def test_langfuse_scores_written(run_agent):
    """Full end-to-end: scores must appear in Langfuse after suite run."""
    from langfuse import Langfuse

    cases = load_golden_cases("evals/fixtures/golden/qa.jsonl")[:2]  # keep cheap
    results = await run_eval_suite(
        agent_fn=run_agent,
        cases=cases,
        metric_factories=[answer_relevancy, task_completion_geval],
    )

    lf = Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )

    for result in results:
        trace = lf.get_trace(result.trace_id)
        scores = {s.name: s.value for s in trace.scores}
        assert "answer_relevancy" in scores, f"Missing score for trace {result.trace_id}"
