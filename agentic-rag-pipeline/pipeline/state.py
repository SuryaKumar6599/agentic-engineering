from typing import TypedDict, List, Annotated
import operator
from typing import Optional
from schemas.models import TaskResult, WorkflowPlan
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State for the multi-agent LangGraph workflow.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    task_context: str
    task: Optional[TaskResult]
    workflow_plan: Optional[WorkflowPlan]
    final_answer: str
    next_step: str
