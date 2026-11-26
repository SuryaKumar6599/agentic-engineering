"""
harness/metrics/task_success.py
────────────────────────────────
Task-success metrics that judge final agent output quality.

Uses DeepEval's built-in LLM-judged metrics + custom wrappers.
When settings.is_local is True all judge calls are routed through
the local OpenAI-compatible server (Ollama / LM Studio / llama.cpp)
using DeepEval's LocalModel wrapper.

  • AnswerRelevancyMetric   – is the response on-topic?
  • FaithfulnessMetric      – is the response grounded in context?
  • GEval (custom rubric)   – flexible LLM-as-judge for task success

All share the same judge model (settings.resolved_model_name).
"""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCaseParams

from harness.config import settings


def _judge_model():
    """
    Return the DeepEval judge model object.

    • Local server  → DeepEval LocalModel pointing at Ollama/LM Studio/llama.cpp
    • Cloud         → model name string (DeepEval resolves via OPENAI_API_KEY)
    """
    if settings.is_local:
        from deepeval.models import LocalModel  # requires deepeval >= 1.4
        return LocalModel(
            model=settings.local_model_name,
            base_url=settings.local_model_base_url,
            api_key=settings.local_model_api_key,
        )
    return settings.eval_model  # plain string → deepeval uses OpenAI


def answer_relevancy(threshold: float = 0.7) -> AnswerRelevancyMetric:
    return AnswerRelevancyMetric(
        threshold=threshold,
        model=_judge_model(),
        include_reason=True,
    )


def faithfulness(threshold: float = 0.7) -> FaithfulnessMetric:
    return FaithfulnessMetric(
        threshold=threshold,
        model=_judge_model(),
        include_reason=True,
    )


def task_completion_geval(
    criteria: str = "The agent fully completed the user's request correctly.",
    threshold: float = 0.7,
) -> GEval:
    """
    Generic LLM-as-judge for task success.
    Evaluates: input (task), actual_output (agent response).
    """
    return GEval(
        name="task_completion",
        criteria=criteria,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
        model=_judge_model(),
    )


def correctness_geval(threshold: float = 0.8) -> GEval:
    """Binary-ish correctness: compares actual vs expected output."""
    return GEval(
        name="correctness",
        criteria=(
            "Given the expected output, evaluate how correct and complete "
            "the actual output is. Penalise hallucinations and omissions."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=_judge_model(),
    )


def ragas_answer_relevancy(threshold: float = 0.7) -> RagasMetric:
    """Ragas answer relevancy bridged into DeepEval."""
    # Ragas always uses the OpenAI client; for local routing set
    # OPENAI_API_KEY + OPENAI_BASE_URL env vars before importing ragas.
    return RagasMetric(
        threshold=threshold,
        model=settings.resolved_model_name,
        metric="answer_relevancy",
    )
