from __future__ import annotations

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from configs.settings import settings
from schemas.models import TaskCategory, TaskResult, WorkflowPlan

_SYSTEM_PROMPT = """\
You are a workflow planning assistant. Given a user query, its classified task type, and optional context,
decompose the work into a minimal ordered list of concrete, actionable steps.

Rules:
- 2 to 6 steps maximum.
- Each step must have a unique integer ID starting from 1.
- The dependencies field maps each step ID (as string) to a list of step IDs (as strings) that must complete first.
- Step 1 never has dependencies.
- Dependencies must not be circular.
- estimated_complexity must be one of: low, medium, high.

Return a structured WorkflowPlan.
"""

_HUMAN_PROMPT = """\
Query: {query}
Task category: {category}
Context: {context}
"""


class WorkflowSegmenter:
    """Decompose a query into a DAG-validated workflow plan.

    Uses .with_structured_output(WorkflowPlan) for guaranteed JSON structure,
    then validates the dependency graph with networkx before returning.
    Retries once on DAG validation failure.
    """

    def __init__(self) -> None:
        base_llm = ChatOpenAI(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            temperature=0.1,
        )
        self._llm = base_llm.with_structured_output(WorkflowPlan)

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])

        self._chain = self._prompt | self._llm

    def segment(self, query: str, task: TaskResult, context: str = "") -> WorkflowPlan:
        """Return a validated, acyclic WorkflowPlan. Retries once on DAG failure."""
        payload = {
            "query": query,
            "category": task.category.value,
            "context": context or "(none)",
        }

        plan: WorkflowPlan = self._chain.invoke(payload)  # type: ignore[assignment]
        try:
            return self._validate_dag(plan)
        except ValueError:
            # Single retry — ask the LLM to fix the dependency structure.
            plan = self._chain.invoke(payload)  # type: ignore[assignment]
            return self._validate_dag(plan)

    # ------------------------------------------------------------------
    # DAG validation (FIX: lack-of-dependency-validation)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dag(plan: WorkflowPlan) -> WorkflowPlan:
        """Raise ValueError if the dependency graph contains self-loops or cycles."""
        G = nx.DiGraph()
        step_ids = {str(s.id) for s in plan.steps}
        G.add_nodes_from(step_ids)

        for step_id, deps in plan.dependencies.items():
            for dep in deps:
                if dep == step_id:
                    raise ValueError(
                        f"Step {step_id} lists itself as a dependency (self-loop)."
                    )
                if dep not in step_ids:
                    raise ValueError(
                        f"Dependency '{dep}' in step {step_id} references a non-existent step."
                    )
                G.add_edge(dep, step_id)

        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            raise ValueError(f"Circular dependency detected in workflow plan: {cycles}")

        return plan
