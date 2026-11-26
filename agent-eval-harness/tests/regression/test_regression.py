"""
tests/regression/test_regression.py
──────────────────────────────────────
Regression suite for CI.

What this tests:
  • Pass rate on the golden dataset must not drop below PASS_RATE_THRESHOLD
  • P95 latency must stay below settings.latency_p95_max_s
  • Total cost must stay below settings.cost_budget_usd

Run with:
    pytest tests/regression/ -m regression --timeout=300
"""

from __future__ import annotations

import statistics

import pytest

from harness.config import settings
from harness.cost import run_cost
from harness.dataset import load_golden_cases
from harness.metrics.task_success import answer_relevancy, task_completion_geval
from harness.models import EvalResult
from harness.runner import run_eval_suite

PASS_RATE_THRESHOLD = 0.80          # 80% of cases must pass
GOLDEN_DATASET = "evals/fixtures/golden/qa.jsonl"


@pytest.fixture(scope="module")
async def suite_results(run_agent) -> list[EvalResult]:
    """Run the full eval suite once per test module."""
    run_cost.reset()
    cases = load_golden_cases(GOLDEN_DATASET)
    return await run_eval_suite(
        agent_fn=run_agent,
        cases=cases,
        metric_factories=[answer_relevancy, task_completion_geval],
    )


@pytest.mark.regression
@pytest.mark.integration
async def test_pass_rate(suite_results):
    """Overall pass rate must meet threshold."""
    passed = sum(1 for r in suite_results if r.passed)
    total = len(suite_results)
    rate = passed / total if total else 0.0
    assert rate >= PASS_RATE_THRESHOLD, (
        f"Pass rate {rate:.1%} < {PASS_RATE_THRESHOLD:.1%} "
        f"({passed}/{total} passed)"
    )


@pytest.mark.regression
@pytest.mark.integration
async def test_latency_p95(suite_results):
    """P95 latency must stay under the configured threshold."""
    latencies = sorted(r.latency_ms / 1000 for r in suite_results)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else latencies[0]
    assert p95 <= settings.latency_p95_max_s, (
        f"P95 latency {p95:.2f}s exceeds limit {settings.latency_p95_max_s}s"
    )


@pytest.mark.regression
@pytest.mark.integration
async def test_cost_budget(suite_results):
    """Total LLM cost must stay within the configured budget."""
    total = run_cost.total
    assert total <= settings.cost_budget_usd, (
        f"Cost ${total:.4f} exceeds budget ${settings.cost_budget_usd:.2f}"
    )


@pytest.mark.regression
@pytest.mark.integration
async def test_no_complete_failures(suite_results):
    """No task should produce ERROR-status metrics (framework/agent crashes)."""
    from harness.models import MetricStatus

    errors = [
        (r.task_id, m.name, m.reason)
        for r in suite_results
        for m in r.metrics
        if m.status == MetricStatus.ERROR
    ]
    assert not errors, f"Tasks with framework errors: {errors}"
