"""
harness/tracer.py
─────────────────
Langfuse tracing wrapper.

Usage
-----
    async with Tracer(task_id="q-001") as t:
        result = await my_agent.run(...)
        t.record(result)          # attach the AgentTrace

Langfuse traces are flushed at the end of the context manager.
When LANGFUSE_SECRET_KEY is empty, the tracer is a no-op.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from harness.config import settings
from harness.models import AgentTrace

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.client import StatefulSpanClient


def _make_client() -> "Langfuse | None":
    key = settings.langfuse_secret_key.strip()
    if not key or key == "sk-lf-..." or key.endswith("..."):
        return None
    from langfuse import Langfuse  # lazy import – not mandatory at startup
    return Langfuse(
        secret_key=key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )


_client: "Langfuse | None" = _make_client()


class Tracer:
    """Context manager that wraps an agent invocation in a Langfuse trace."""

    def __init__(self, task_id: str, session_id: str | None = None):
        self.task_id = task_id
        self.session_id = session_id or task_id
        self._trace_id = (
            _client.create_trace_id()
            if _client and hasattr(_client, "create_trace_id")
            else str(uuid.uuid4())
        )
        self._start: float = 0.0
        self._lf_trace = None
        self._span: "StatefulSpanClient | None" = None

    async def __aenter__(self) -> "Tracer":
        self._start = time.perf_counter()
        if _client:
            if hasattr(_client, "start_observation"):
                # Langfuse v3+/v5 OpenTelemetry SDK
                self._span = _client.start_observation(
                    name=f"eval:{self.task_id}",
                    as_type="span",
                    trace_context={"trace_id": self._trace_id},
                    metadata={"session_id": self.session_id, "tags": ["eval"]},
                )
            elif hasattr(_client, "trace"):
                # Langfuse v2 SDK
                self._lf_trace = _client.trace(
                    id=self._trace_id,
                    name=f"eval:{self.task_id}",
                    session_id=self.session_id,
                    tags=["eval"],
                )
                self._span = self._lf_trace.span(name="agent_run")
        return self

    async def __aexit__(self, *_) -> None:
        if self._span:
            self._span.end()
        if _client:
            _client.flush()

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1_000

    def record_generation(
        self,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """Log an LLM generation inside the current trace span."""
        if _client and hasattr(_client, "start_observation"):
            gen = _client.start_observation(
                name=name,
                as_type="generation",
                trace_context={"trace_id": self._trace_id},
                model=model,
                input=prompt,
                output=completion,
                usage_details={"input": prompt_tokens, "output": completion_tokens},
                metadata={"cost_usd": cost_usd},
            )
            gen.end()
        elif self._lf_trace:
            self._lf_trace.generation(
                name=name,
                model=model,
                input=prompt,
                output=completion,
                usage={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "unit": "TOKENS",
                },
                metadata={"cost_usd": cost_usd},
            )

    def record_tool_call(self, name: str, args: dict, result) -> None:
        """Log a tool call event inside the current trace."""
        if _client and hasattr(_client, "start_observation"):
            span = _client.start_observation(
                name=f"tool:{name}",
                as_type="tool",
                trace_context={"trace_id": self._trace_id},
                input=args,
                output={"result": str(result)[:500]},
            )
            span.end()
        elif self._lf_trace:
            self._lf_trace.span(
                name=f"tool:{name}",
                input=args,
                output={"result": str(result)[:500]},
            )

    def add_score(self, name: str, value: float, comment: str = "") -> None:
        """Attach a numeric eval score to the trace (visible in Langfuse UI)."""
        if not _client:
            return
        if hasattr(_client, "create_score"):
            _client.create_score(
                trace_id=self._trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        elif self._lf_trace and hasattr(_client, "score"):
            _client.score(
                trace_id=self._trace_id,
                name=name,
                value=value,
                comment=comment,
            )
