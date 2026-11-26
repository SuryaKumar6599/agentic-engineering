"""
harness/runner.py
─────────────────
The core eval runner.  Given an agent callable, a list of GoldenCases, and
a metric factory, it:

  1. Runs the agent (with Langfuse tracing)
  2. Measures all metrics (DeepEval + custom)
  3. Checks cost + latency budgets
  4. Returns a list of EvalResult objects
  5. Pushes scores back to Langfuse

Usage
-----
    results = await run_eval_suite(
        agent_fn=my_agent,
        cases=load_golden_cases("evals/fixtures/golden/qa.jsonl"),
        metric_factories=[answer_relevancy, faithfulness, task_completion_geval],
    )
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from typing import Any, Awaitable, Callable

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from rich.console import Console
from rich.table import Table

from harness.config import settings
from harness.cost import cost_usd, run_cost
from harness.metrics.tool_accuracy import build_tool_metrics
from harness.models import (
    AgentTrace,
    EvalResult,
    GoldenCase,
    MetricResult,
    MetricStatus,
)
from harness.tracer import Tracer

console = Console()

AgentFn = Callable[[str], Awaitable[AgentTrace]]


async def _run_one(
    agent_fn: AgentFn,
    case: GoldenCase,
    metric_factories: list[Callable],
    sem: asyncio.Semaphore,
) -> EvalResult:
    """Run a single eval case end-to-end."""
    async with sem:
        async with Tracer(task_id=case.task_id) as tracer:
            # ── 1. invoke agent ────────────────────────────────────────────
            trace: AgentTrace = await agent_fn(case.input)
            trace.trace_id = tracer.trace_id
            trace.task_id = case.task_id

            # ── 2. accumulate cost ─────────────────────────────────────────
            call_cost = cost_usd(trace.model, trace.prompt_tokens, trace.completion_tokens)
            trace.cost_usd = call_cost
            run_cost.add(call_cost)

            # ── 3. build DeepEval test case ────────────────────────────────
            test_case = LLMTestCase(
                input=case.input,
                actual_output=trace.output,
                expected_output=case.expected_output or "",
                retrieval_context=[],   # populate for RAG agents
            )

            # ── 4. run LLM-judged metrics ──────────────────────────────────
            metrics = [f() for f in metric_factories]
            metric_results: list[MetricResult] = []

            for metric in metrics:
                metric_name = (
                    getattr(metric, "name", None)
                    or getattr(metric, "__name__", None)
                    or type(metric).__name__
                )
                try:
                    await metric.a_measure(test_case)
                    status = MetricStatus.PASS if metric.is_successful() else MetricStatus.FAIL
                    mr = MetricResult(
                        name=metric_name,
                        score=float(getattr(metric, "score", 0) or 0),
                        threshold=float(getattr(metric, "threshold", 0) or 0),
                        status=status,
                        reason=getattr(metric, "reason", None) or "",
                    )
                except Exception as exc:
                    mr = MetricResult(
                        name=metric_name,
                        status=MetricStatus.ERROR,
                        reason=str(exc),
                    )
                metric_results.append(mr)
                # Push score to Langfuse
                if mr.score is not None:
                    tracer.add_score(mr.name, mr.score, mr.reason)

            # ── 5. deterministic tool-call metrics ────────────────────────
            for tool_metric in build_tool_metrics(case, trace):
                tool_metric.measure(test_case)
                status = MetricStatus.PASS if tool_metric.is_successful() else MetricStatus.FAIL
                mr = MetricResult(
                    name=tool_metric.name,
                    score=float(getattr(tool_metric, "score", 0) or 0),
                    threshold=float(getattr(tool_metric, "threshold", 1.0) or 1.0),
                    status=status,
                    reason=getattr(tool_metric, "reason", None) or "",
                )
                metric_results.append(mr)
                if mr.score is not None:
                    tracer.add_score(mr.name, mr.score, mr.reason)

            # ── 6. latency metric ─────────────────────────────────────────
            latency_ms = tracer.elapsed_ms
            latency_ok = (latency_ms / 1000) <= settings.latency_p95_max_s
            metric_results.append(
                MetricResult(
                    name="latency_ok",
                    score=latency_ms / 1000,
                    threshold=settings.latency_p95_max_s,
                    status=MetricStatus.PASS if latency_ok else MetricStatus.FAIL,
                    reason=f"{latency_ms:.0f} ms",
                )
            )

            return EvalResult(
                task_id=case.task_id,
                trace_id=tracer.trace_id,
                model=trace.model,
                judge_model=settings.resolved_model_name,
                metrics=metric_results,
                total_cost_usd=call_cost,
                latency_ms=latency_ms,
            )


async def run_eval_suite(
    agent_fn: AgentFn,
    cases: list[GoldenCase],
    metric_factories: list[Callable] | None = None,
) -> list[EvalResult]:
    """
    Run all eval cases concurrently (bounded by EVAL_MAX_CONCURRENCY).
    Returns list of EvalResult, one per case.
    """
    metric_factories = metric_factories or []
    sem = asyncio.Semaphore(settings.eval_max_concurrency)
    tasks = [
        _run_one(agent_fn, case, metric_factories, sem)
        for case in cases
    ]
    results: list[EvalResult] = await asyncio.gather(*tasks)

    # ── budget gate ────────────────────────────────────────────────────────────
    total_cost = run_cost.total
    if total_cost > settings.cost_budget_usd:
        console.print(
            f"[bold red]⛔ COST BUDGET EXCEEDED: ${total_cost:.4f} > "
            f"${settings.cost_budget_usd:.2f}[/bold red]"
        )

    _print_summary(results, total_cost)
    return results


# ── pretty-print ───────────────────────────────────────────────────────────────

def _print_summary(results: list[EvalResult], total_cost: float) -> None:
    table = Table(title="Eval Suite Summary", show_lines=True)
    table.add_column("Task ID", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Pass?", justify="center")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Failing Metrics")

    for r in results:
        failures = [
            f"{m.name} ({m.status.value})" if m.status == MetricStatus.ERROR else m.name
            for m in r.metrics
            if m.status in (MetricStatus.FAIL, MetricStatus.ERROR)
        ]
        table.add_row(
            r.task_id,
            r.model or "—",
            "✅" if r.passed else "❌",
            f"${r.total_cost_usd:.4f}",
            f"{r.latency_ms:.0f}",
            ", ".join(failures) or "—",
        )

    console.print(table)

    passed = sum(1 for r in results if r.passed)
    latencies = [r.latency_ms for r in results]
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else (latencies[0] if latencies else 0)
    console.print(
        f"\n[bold]Pass rate:[/bold] {passed}/{len(results)} "
        f"[bold]  Total cost:[/bold] ${total_cost:.4f} "
        f"[bold]  P95 latency:[/bold] {p95:.0f} ms"
    )
