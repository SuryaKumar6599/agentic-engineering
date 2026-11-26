from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from configs.settings import settings
from pipeline.state import AgentState
from pipeline.task_detector import TaskDetector
from pipeline.workflow_segmenter import WorkflowSegmenter
from tools.rag_tools import vector_search, summarize_context, web_search
import json

class Route(BaseModel):
    next_node: Literal["researcher", "planner", "synthesizer", "FINISH"]


def supervisor_node(state: AgentState) -> dict:
    # Deterministic router to optimize time complexity.
    if not state.get('task_context'):
        next_step = "researcher"
    elif not state.get('workflow_plan'):
        next_step = "planner"
    elif not state.get('final_answer'):
        next_step = "synthesizer"
    else:
        next_step = "FINISH"
            
    return {"next_step": next_step, "messages": [AIMessage(content=f"Supervisor: Handing off to {next_step}...")]}


def researcher_node(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
        temperature=0.2,
    )
    
    # Bind tools so the LLM can decide which to use
    llm_with_tools = llm.bind_tools([vector_search, web_search])
    
    sys_msg = SystemMessage(content=(
        "You are a Research Agent. You must gather context to answer the user's query.\n"
        "You have access to two tools:\n"
        "1. vector_search: Use this FIRST for questions about our internal ML project documentation or REST APIs.\n"
        "2. web_search: Use this for general knowledge, latest news, or if the user's query is clearly not about our local docs.\n"
        "Choose the most appropriate tool and call it."
    ))
    
    response = llm_with_tools.invoke([sys_msg, HumanMessage(content=state["query"])])
    
    docs_json = "[]"
    used_tool = "None"
    
    # If the LLM successfully decided to call a tool
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "vector_search":
            docs_json = vector_search.invoke(tool_call["args"])
            used_tool = "Local Vector Search"
        elif tool_call["name"] == "web_search":
            docs_json = web_search.invoke(tool_call["args"])
            used_tool = "Web Search"
    else:
        # Fallback just in case the LLM didn't call a tool
        docs_json = vector_search.invoke({"query": state["query"]})
        used_tool = "Local Vector Search (Fallback)"
        
    msg = f"Researcher: Gathered context using {used_tool}."
        
    return {
        "task_context": docs_json,
        "messages": [AIMessage(content=msg)]
    }


def planner_node(state: AgentState) -> dict:
    detector = TaskDetector()
    segmenter = WorkflowSegmenter()
    query = state["query"]
    context = state.get("task_context", "")
    
    task_res = detector.detect(query, context)
    plan = segmenter.segment(query, task_res, context)
    
    return {
        "task": task_res,
        "workflow_plan": plan,
        "messages": [AIMessage(content="Planner: Task classification and workflow segmentation complete.")]
    }


def synthesizer_node(state: AgentState) -> dict:
    context = state.get("task_context", "")
    task = state.get("task")
    category = task.category.value if task else "GENERAL_QA"
    
    input_text = (
        f"Query: {state['query']}\n"
        f"Category: {category}\n"
        f"Workflow Plan: {state.get('workflow_plan')}\n"
        f"Context: {context}"
    )
    summary = summarize_context.invoke({"input_text": input_text})
    
    return {
        "final_answer": summary,
        "messages": [AIMessage(content="Synthesizer: Final answer generated.")]
    }


def router(state: AgentState) -> str:
    return state["next_step"]


def create_multiagent_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("planner", planner_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    builder.add_edge(START, "supervisor")
    
    # Conditional routing from supervisor
    builder.add_conditional_edges("supervisor", router, {
        "researcher": "researcher",
        "planner": "planner",
        "synthesizer": "synthesizer",
        "FINISH": END
    })
    
    # All workers report back to the supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("planner", "supervisor")
    builder.add_edge("synthesizer", "supervisor")
    
    return builder.compile()
