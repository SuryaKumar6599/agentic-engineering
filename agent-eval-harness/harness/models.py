"""
harness/models.py
─────────────────
Shared Pydantic models used across all eval runners.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Tool-call models ────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: Any = None


class AgentTrace(BaseModel):
    """Full trace of one agent invocation."""
    trace_id: str
    task_id: str
    input: str
    output: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Golden-dataset models ────────────────────────────────────────────────────────

class ExpectedToolCall(BaseModel):
    name: str
    required_args: list[str] = Field(default_factory=list)
    forbidden_args: list[str] = Field(default_factory=list)


class GoldenCase(BaseModel):
    """One entry in the golden regression dataset."""
    task_id: str
    input: str
    expected_output: str | None = None          # for exact/fuzzy match
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    expected_tool_order: bool = False            # True → sequence must match
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Metric result models ────────────────────────────────────────────────────────

class MetricStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class MetricResult(BaseModel):
    name: str
    score: float | None = None
    threshold: float | None = None
    status: MetricStatus
    reason: str = ""


class EvalResult(BaseModel):
    task_id: str
    trace_id: str
    model: str = ""
    judge_model: str = ""
    metrics: list[MetricResult]
    total_cost_usd: float = 0.0
    latency_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return all(m.status == MetricStatus.PASS for m in self.metrics)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "judge_model": self.judge_model,
            "passed": self.passed,
            "cost_usd": self.total_cost_usd,
            "latency_ms": self.latency_ms,
            "metrics": [m.model_dump() for m in self.metrics],
        }
