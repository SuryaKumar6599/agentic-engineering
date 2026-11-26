from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.models import TaskCategory, TaskResult, WorkflowPlan, Step


class TestTaskResult:
    def test_valid_construction(self):
        t = TaskResult(
            category=TaskCategory.CODE_GENERATION,
            confidence=0.85,
            rationale="Test rationale",
        )
        assert t.category == TaskCategory.CODE_GENERATION
        assert t.confidence == 0.85

    def test_confidence_bounds_high(self):
        with pytest.raises(ValidationError):
            TaskResult(category=TaskCategory.RESEARCH, confidence=1.1, rationale="x")

    def test_confidence_bounds_low(self):
        with pytest.raises(ValidationError):
            TaskResult(category=TaskCategory.RESEARCH, confidence=-0.1, rationale="x")

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            TaskResult(category="UNKNOWN_CATEGORY", confidence=0.5, rationale="x")


class TestWorkflowPlan:
    def test_empty_dependencies_default(self):
        plan = WorkflowPlan(steps=[Step(id=1, description="Do something")])
        assert plan.dependencies == {}

    def test_step_id_required(self):
        with pytest.raises(ValidationError):
            Step(description="No ID provided")

    def test_full_plan_construction(self):
        plan = WorkflowPlan(
            steps=[
                Step(id=1, description="Setup", estimated_complexity="low"),
                Step(id=2, description="Implement", estimated_complexity="high"),
            ],
            dependencies={"2": ["1"]},
        )
        assert len(plan.steps) == 2
        assert plan.dependencies["2"] == ["1"]
