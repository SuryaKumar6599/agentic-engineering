from __future__ import annotations

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field


class TaskCategory(str, Enum):
    """Canonical task taxonomy. Extend by adding new members."""

    RESEARCH = "RESEARCH"
    CODE_GENERATION = "CODE_GENERATION"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    PLANNING = "PLANNING"
    GENERAL_QA = "GENERAL_QA"


class TaskResult(BaseModel):
    """Structured output from the task classification LLM call."""

    category: TaskCategory
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence 0–1")
    rationale: str = Field(description="One-sentence explanation of the classification decision")


class Step(BaseModel):
    """A single atomic step in a workflow plan."""

    id: int = Field(description="1-based sequential step identifier")
    description: str = Field(description="Clear, actionable description of what this step does")
    estimated_complexity: str = Field(
        default="medium",
        description="Rough complexity: low | medium | high",
    )


class WorkflowPlan(BaseModel):
    """Structured, DAG-validated workflow plan returned by the segmenter."""

    steps: List[Step] = Field(description="Ordered list of workflow steps")
    dependencies: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Adjacency list mapping step_id -> list of step_ids that must complete first. "
            "Example: {'2': ['1'], '3': ['1', '2']}. Keys and values are string step IDs."
        ),
    )


class AgentResponse(BaseModel):
    """Top-level response envelope returned by AgenticRAGAgent.run()."""

    answer: str
    task: TaskResult
    workflow: WorkflowPlan
    sources: List[str] = Field(default_factory=list, description="Source document excerpts used")
    mode: str = Field(default="rag", description="'rag' or 'zero_shot'")
