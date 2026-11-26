"""
harness/metrics/tool_accuracy.py
─────────────────────────────────
Tool-call accuracy metrics:

  • ToolCallPresence  – did the agent call the right tools?
  • ToolCallOrder     – were they called in the right sequence?
  • ToolArgCoverage   – were required args present / forbidden args absent?
  • ToolHallucinationRate – did the agent call tools that weren't expected?

All metrics implement DeepEval's BaseMetric interface so they slot into
the standard deepeval.evaluate() pipeline unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from harness.models import AgentTrace, GoldenCase, MetricResult, MetricStatus


# ── helpers ────────────────────────────────────────────────────────────────────

def _called_names(trace: AgentTrace) -> list[str]:
    return [tc.name for tc in trace.tool_calls]


# ── Metric: tool presence ──────────────────────────────────────────────────────

@dataclass
class ToolCallPresenceMetric(BaseMetric):
    """Fraction of expected tools that were actually called."""

    threshold: float = 1.0
    name: str = "tool_call_presence"
    golden: GoldenCase | None = None
    trace: AgentTrace | None = None

    def measure(self, test_case: LLMTestCase) -> float:  # type: ignore[override]
        if not self.golden or not self.golden.expected_tool_calls:
            self.score = 1.0
            self.success = True
            return self.score

        called = set(_called_names(self.trace))
        expected = {e.name for e in self.golden.expected_tool_calls}
        self.score = len(called & expected) / len(expected)
        self.success = self.score >= self.threshold
        self.reason = (
            f"Called {called & expected} of {expected}"
            if not self.success
            else "All expected tools called"
        )
        return self.score

    def is_successful(self) -> bool:
        return bool(self.success)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # type: ignore[override]
        return self.measure(test_case)


# ── Metric: tool order ─────────────────────────────────────────────────────────

@dataclass
class ToolCallOrderMetric(BaseMetric):
    """Binary: 1.0 if tool call sequence matches expected order, else 0."""

    threshold: float = 1.0
    name: str = "tool_call_order"
    golden: GoldenCase | None = None
    trace: AgentTrace | None = None

    def measure(self, test_case: LLMTestCase) -> float:  # type: ignore[override]
        if not self.golden or not self.golden.expected_tool_order:
            self.score = 1.0
            self.success = True
            return self.score

        called = _called_names(self.trace)
        expected = [e.name for e in self.golden.expected_tool_calls]
        # subsequence check: expected must appear in order inside called
        idx = 0
        for name in called:
            if idx < len(expected) and name == expected[idx]:
                idx += 1
        self.score = 1.0 if idx == len(expected) else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"Expected order {expected}, got {called}"
            if not self.success
            else "Tool call order correct"
        )
        return self.score

    def is_successful(self) -> bool:
        return bool(self.success)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # type: ignore[override]
        return self.measure(test_case)


# ── Metric: argument coverage ──────────────────────────────────────────────────

@dataclass
class ToolArgCoverageMetric(BaseMetric):
    """
    For each expected tool call:
      - required_args must be present in actual args
      - forbidden_args must NOT be present
    Score = fraction of tool calls fully compliant.
    """

    threshold: float = 1.0
    name: str = "tool_arg_coverage"
    golden: GoldenCase | None = None
    trace: AgentTrace | None = None

    def measure(self, test_case: LLMTestCase) -> float:  # type: ignore[override]
        if not self.golden or not self.golden.expected_tool_calls:
            self.score = 1.0
            self.success = True
            return self.score

        actual_by_name: dict[str, list[dict]] = {}
        for tc in self.trace.tool_calls:
            actual_by_name.setdefault(tc.name, []).append(tc.arguments)

        compliant = 0
        total = len(self.golden.expected_tool_calls)
        failures: list[str] = []

        for exp in self.golden.expected_tool_calls:
            calls = actual_by_name.get(exp.name, [])
            if not calls:
                failures.append(f"{exp.name} not called")
                continue
            # At least one call must satisfy all constraints
            ok = any(
                all(r in c for r in exp.required_args)
                and all(f not in c for f in exp.forbidden_args)
                for c in calls
            )
            if ok:
                compliant += 1
            else:
                failures.append(f"{exp.name} arg constraints violated")

        self.score = compliant / total if total else 1.0
        self.success = self.score >= self.threshold
        self.reason = "; ".join(failures) if failures else "All arg constraints met"
        return self.score

    def is_successful(self) -> bool:
        return bool(self.success)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # type: ignore[override]
        return self.measure(test_case)


# ── Metric: hallucinated tools ─────────────────────────────────────────────────

@dataclass
class ToolHallucinationRateMetric(BaseMetric):
    """
    Fraction of called tools that were NOT in the expected set.
    Score = 1 - hallucination_rate  (higher is better).
    """

    threshold: float = 1.0
    name: str = "tool_hallucination_rate"
    golden: GoldenCase | None = None
    trace: AgentTrace | None = None

    def measure(self, test_case: LLMTestCase) -> float:  # type: ignore[override]
        if not self.golden or not self.trace.tool_calls:
            self.score = 1.0
            self.success = True
            return self.score

        called = _called_names(self.trace)
        expected = {e.name for e in self.golden.expected_tool_calls}
        hallucinated = [t for t in called if t not in expected]
        hall_rate = len(hallucinated) / len(called) if called else 0.0
        self.score = 1.0 - hall_rate
        self.success = self.score >= self.threshold
        self.reason = (
            f"Hallucinated tools: {hallucinated}"
            if hallucinated
            else "No hallucinated tools"
        )
        return self.score

    def is_successful(self) -> bool:
        return bool(self.success)

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # type: ignore[override]
        return self.measure(test_case)


# ── Convenience: build all tool metrics for a case ────────────────────────────

def build_tool_metrics(
    golden: GoldenCase,
    trace: AgentTrace,
) -> list[BaseMetric]:
    shared = dict(golden=golden, trace=trace)
    return [
        ToolCallPresenceMetric(**shared),
        ToolCallOrderMetric(**shared),
        ToolArgCoverageMetric(**shared),
        ToolHallucinationRateMetric(**shared),
    ]
