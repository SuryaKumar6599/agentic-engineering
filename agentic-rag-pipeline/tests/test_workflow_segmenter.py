from __future__ import annotations

import pytest

from schemas.models import Step, WorkflowPlan, TaskCategory, TaskResult
from pipeline.workflow_segmenter import WorkflowSegmenter


def _make_task(category: TaskCategory = TaskCategory.CODE_GENERATION) -> TaskResult:
    return TaskResult(category=category, confidence=0.9, rationale="test")


def _make_plan(steps: list[dict], deps: dict) -> WorkflowPlan:
    return WorkflowPlan(
        steps=[Step(**s) for s in steps],
        dependencies=deps,
    )


class TestDAGValidation:
    """WorkflowSegmenter._validate_dag() must catch all invalid graphs."""

    def test_valid_linear_dag(self):
        plan = _make_plan(
            steps=[{"id": 1, "description": "A"}, {"id": 2, "description": "B"}],
            deps={"2": ["1"]},
        )
        result = WorkflowSegmenter._validate_dag(plan)
        assert len(result.steps) == 2

    def test_valid_diamond_dag(self):
        plan = _make_plan(
            steps=[
                {"id": 1, "description": "A"},
                {"id": 2, "description": "B"},
                {"id": 3, "description": "C"},
                {"id": 4, "description": "D"},
            ],
            deps={"2": ["1"], "3": ["1"], "4": ["2", "3"]},
        )
        WorkflowSegmenter._validate_dag(plan)  # should not raise

    def test_self_loop_raises(self):
        plan = _make_plan(
            steps=[{"id": 1, "description": "A"}, {"id": 2, "description": "B"}],
            deps={"2": ["2"]},  # self-loop
        )
        with pytest.raises(ValueError, match="itself as a dependency"):
            WorkflowSegmenter._validate_dag(plan)

    def test_direct_cycle_raises(self):
        plan = _make_plan(
            steps=[{"id": 1, "description": "A"}, {"id": 2, "description": "B"}],
            deps={"2": ["1"], "1": ["2"]},  # 1 -> 2 -> 1
        )
        with pytest.raises(ValueError, match="Circular dependency"):
            WorkflowSegmenter._validate_dag(plan)

    def test_three_node_cycle_raises(self):
        plan = _make_plan(
            steps=[
                {"id": 1, "description": "A"},
                {"id": 2, "description": "B"},
                {"id": 3, "description": "C"},
            ],
            deps={"2": ["1"], "3": ["2"], "1": ["3"]},  # 1->2->3->1
        )
        with pytest.raises(ValueError, match="Circular dependency"):
            WorkflowSegmenter._validate_dag(plan)

    def test_nonexistent_dependency_raises(self):
        plan = _make_plan(
            steps=[{"id": 1, "description": "A"}],
            deps={"1": ["99"]},  # step 99 doesn't exist
        )
        with pytest.raises(ValueError, match="non-existent step"):
            WorkflowSegmenter._validate_dag(plan)

    def test_no_dependencies_is_valid(self):
        plan = _make_plan(
            steps=[{"id": 1, "description": "A"}, {"id": 2, "description": "B"}],
            deps={},
        )
        result = WorkflowSegmenter._validate_dag(plan)
        assert result.dependencies == {}
