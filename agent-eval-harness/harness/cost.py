"""
harness/cost.py
───────────────
Per-token cost table + running accumulator.

Local models (Ollama / LM Studio / llama.cpp) have $0 API cost.
Any model name containing "qwen", "llama", "mistral", "phi", or "gemma"
that is served from a local endpoint is treated as free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

# Cost in USD per 1 000 tokens (input, output)
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":               (0.005,  0.015),
    "gpt-4o-mini":          (0.00015, 0.0006),
    "gpt-4-turbo":          (0.010,  0.030),
    "gpt-3.5-turbo":        (0.0005, 0.0015),
    "o1-preview":           (0.015,  0.060),
    # Anthropic
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022":  (0.0008, 0.004),
    "claude-3-opus-20240229":     (0.015, 0.075),
    # ── Local / self-hosted → always $0.00 ─────────────────────────────────────
    "qwen3.6:27b":               (0.0, 0.0),
    "qwen2.5:7b":                (0.0, 0.0),
    "llama3.2:3b":               (0.0, 0.0),
    "mistral:latest":            (0.0, 0.0),
    "qwen3:30b-a3b-ud-q2_k_xl":  (0.0, 0.0),
    "qwen3":                     (0.0, 0.0),
}

# Prefixes whose cost is always zero (local open-weight models)
_LOCAL_FREE_PREFIXES = ("qwen", "llama", "mistral", "phi", "gemma", "deepseek", "falcon")


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return cost in USD for a single call. Local models are always $0."""
    key = model.lower().split("/")[-1]  # strip provider prefix e.g. "ollama/"
    # Explicit table match first
    if key in _PRICE_TABLE:
        prices = _PRICE_TABLE[key]
    # Heuristic: local open-weight model names
    elif any(key.startswith(p) for p in _LOCAL_FREE_PREFIXES):
        prices = (0.0, 0.0)
    else:
        prices = (0.005, 0.015)  # conservative default
    return (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000


@dataclass
class CostAccumulator:
    """Thread-safe running total across all eval traces."""
    _total: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def add(self, amount: float) -> None:
        with self._lock:
            self._total += amount

    @property
    def total(self) -> float:
        with self._lock:
            return self._total

    def reset(self) -> None:
        with self._lock:
            self._total = 0.0


# Singleton used by all runners in a process
run_cost = CostAccumulator()


@dataclass
class CostAccumulator:
    """Thread-safe running total across all eval traces."""
    _total: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def add(self, amount: float) -> None:
        with self._lock:
            self._total += amount

    @property
    def total(self) -> float:
        with self._lock:
            return self._total

    def reset(self) -> None:
        with self._lock:
            self._total = 0.0


# Singleton used by all runners in a process
run_cost = CostAccumulator()
