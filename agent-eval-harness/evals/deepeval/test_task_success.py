"""
evals/deepeval/test_task_success.py
─────────────────────────────────────
DeepEval-based task-success tests.

Run:
    pytest evals/deepeval/test_task_success.py -m integration
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from harness.dataset import load_golden_cases
from harness.metrics.task_success import answer_relevancy, correctness_geval, task_completion_geval

CASES = load_golden_cases("evals/fixtures/golden/qa.jsonl")


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=[c.task_id for c in CASES])
async def test_answer_relevancy(case, run_agent):
    """Agent output must be relevant to the input."""
    trace = await run_agent(case.input)
    test_case = LLMTestCase(input=case.input, actual_output=trace.output)
    assert_test(test_case, [answer_relevancy(threshold=0.6)])


@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c.expected_output],
    ids=[c.task_id for c in CASES if c.expected_output],
)
async def test_correctness(case, run_agent):
    """When a golden answer exists, output must be correct."""
    trace = await run_agent(case.input)
    test_case = LLMTestCase(
        input=case.input,
        actual_output=trace.output,
        expected_output=case.expected_output,
    )
    assert_test(test_case, [correctness_geval(threshold=0.75)])


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=[c.task_id for c in CASES])
async def test_task_completion(case, run_agent):
    """Agent must fully complete the user's request."""
    trace = await run_agent(case.input)
    test_case = LLMTestCase(input=case.input, actual_output=trace.output)
    assert_test(test_case, [task_completion_geval(threshold=0.7)])
