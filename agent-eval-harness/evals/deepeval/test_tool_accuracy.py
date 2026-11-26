"""
evals/deepeval/test_tool_accuracy.py
──────────────────────────────────────
Deterministic tool-call accuracy tests (no LLM judge needed).

Run:
    pytest evals/deepeval/test_tool_accuracy.py -m unit
"""

from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from harness.dataset import load_golden_cases, filter_by_tag
from harness.metrics.tool_accuracy import (
    ToolArgCoverageMetric,
    ToolCallOrderMetric,
    ToolCallPresenceMetric,
    ToolHallucinationRateMetric,
    build_tool_metrics,
)
from harness.models import AgentTrace, ToolCall

ALL_CASES = load_golden_cases("evals/fixtures/golden/qa.jsonl")
TOOL_CASES = [c for c in ALL_CASES if c.expected_tool_calls]


@pytest.mark.integration
@pytest.mark.parametrize("case", TOOL_CASES, ids=[c.task_id for c in TOOL_CASES])
async def test_tool_presence(case, run_agent):
    trace = await run_agent(case.input)
    metric = ToolCallPresenceMetric(golden=case, trace=trace)
    test_case = LLMTestCase(input=case.input, actual_output=trace.output)
    metric.measure(test_case)
    assert metric.is_successful(), metric.reason


@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    [c for c in TOOL_CASES if c.expected_tool_order],
    ids=[c.task_id for c in TOOL_CASES if c.expected_tool_order],
)
async def test_tool_order(case, run_agent):
    trace = await run_agent(case.input)
    metric = ToolCallOrderMetric(golden=case, trace=trace)
    test_case = LLMTestCase(input=case.input, actual_output=trace.output)
    metric.measure(test_case)
    assert metric.is_successful(), metric.reason


@pytest.mark.integration
@pytest.mark.parametrize("case", TOOL_CASES, ids=[c.task_id for c in TOOL_CASES])
async def test_no_hallucinated_tools(case, run_agent):
    trace = await run_agent(case.input)
    metric = ToolHallucinationRateMetric(golden=case, trace=trace)
    test_case = LLMTestCase(input=case.input, actual_output=trace.output)
    metric.measure(test_case)
    assert metric.is_successful(), metric.reason


# ── Offline unit tests (no agent invocation) ──────────────────────────────────

@pytest.mark.unit
def test_tool_presence_offline():
    """Pure unit test: verify metric logic with hand-crafted trace."""
    from harness.models import GoldenCase, ExpectedToolCall

    golden = GoldenCase(
        task_id="offline-001",
        input="test",
        expected_tool_calls=[ExpectedToolCall(name="calculator", required_args=["expression"])],
    )
    trace = AgentTrace(
        trace_id="t1", task_id="offline-001", input="test", output="ok",
        tool_calls=[ToolCall(name="calculator", arguments={"expression": "1+1"}, result="2")],
        latency_ms=100,
    )
    metric = ToolCallPresenceMetric(golden=golden, trace=trace)
    test_case = LLMTestCase(input="test", actual_output="ok")
    metric.measure(test_case)
    assert metric.is_successful()


@pytest.mark.unit
def test_tool_hallucination_offline():
    """Agent called an unexpected tool → hallucination rate = 1."""
    from harness.models import GoldenCase, ExpectedToolCall

    golden = GoldenCase(
        task_id="offline-002",
        input="test",
        expected_tool_calls=[ExpectedToolCall(name="calculator")],
    )
    trace = AgentTrace(
        trace_id="t2", task_id="offline-002", input="test", output="ok",
        tool_calls=[ToolCall(name="send_email", arguments={}, result="sent")],
        latency_ms=100,
    )
    metric = ToolHallucinationRateMetric(golden=golden, trace=trace)
    test_case = LLMTestCase(input="test", actual_output="ok")
    metric.measure(test_case)
    assert not metric.is_successful()
