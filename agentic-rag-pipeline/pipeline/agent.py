from __future__ import annotations

import json
import textwrap
from typing import Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from configs.settings import settings
from pipeline.retriever import FAISSRetriever, get_embedding_model
from pipeline.task_detector import TaskDetector
from pipeline.workflow_segmenter import WorkflowSegmenter
from schemas.models import AgentResponse, TaskResult, WorkflowPlan
from tools.rag_tools import classify_task, summarize_context, vector_search
from tools.rag_tools import set_retriever as _set_retriever


class AgenticRAGAgent:
    """True ReAct agent over three tools: vector_search, classify_task, summarize_context.

    The LLM drives its own execution loop — it decides when retrieved context is
    sufficient and whether to re-query FAISS before classifying. After the agent
    loop produces a final answer, task classification and DAG-validated workflow
    segmentation run as deterministic post-processing steps.

    Parameters
    ----------
    retriever:
        Optional custom retriever (e.g. for tests). Defaults to the process singleton.
    """

    from tools.rag_tools import vector_search, classify_task, summarize_context, web_search
    _TOOLS = [vector_search, classify_task, summarize_context, web_search]

    def __init__(self, retriever: Optional[FAISSRetriever] = None) -> None:
        _retriever = retriever or FAISSRetriever(embeddings=get_embedding_model())
        _set_retriever(_retriever)  # wire into tool singletons
        self._retriever = _retriever

        self._detector = TaskDetector()
        self._segmenter = WorkflowSegmenter()

        llm = ChatOpenAI(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            temperature=0.2,
        )

        # Pull the standard ReAct prompt from LangChain Hub.
        # Format: Thought / Action / Action Input / Observation / ... / Final Answer
        react_prompt = PromptTemplate.from_template(
            "Answer the following questions as best you can. You have access to the following tools:\n\n"
            "{tools}\n\n"
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\n"
            "Question: {input}\n"
            "Thought:{agent_scratchpad}"
        )

        agent = create_react_agent(llm=llm, tools=self._TOOLS, prompt=react_prompt)

        self._executor = AgentExecutor(
            agent=agent,
            tools=self._TOOLS,
            max_iterations=settings.agent_max_iterations,
            verbose=settings.agent_verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> AgentResponse:
        """Execute the full agentic pipeline and return a structured AgentResponse."""
        mode = "rag" if self._retriever.is_loaded else "zero_shot"

        if mode == "zero_shot":
            context = ""
            agent_input = self._build_zero_shot_input(query)
        else:
            # Pre-fetch one pass of context so the agent has something to reason about.
            # The agent may still call vector_search again with a refined query.
            pre_docs = self._retriever.retrieve(query)
            context = "\n\n---\n\n".join(d.page_content for d in pre_docs)
            agent_input = self._build_rag_input(query, context)

        result = self._executor.invoke(agent_input)
        agent_answer: str = result.get("output", "")

        # -- Post-processing: deterministic structured classification + planning --
        task: TaskResult = self._detector.detect(query=query, context=context)
        workflow: WorkflowPlan = self._segmenter.segment(
            query=query, task=task, context=context
        )

        sources = self._extract_sources(result.get("intermediate_steps", []))

        return AgentResponse(
            answer=agent_answer,
            task=task,
            workflow=workflow,
            sources=sources,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rag_input(query: str, context: str) -> dict:
        return {
            "input": textwrap.dedent(f"""\
                Use the available tools to answer the following query accurately.
                Start by calling vector_search to retrieve relevant context.
                If the returned context seems insufficient or off-topic, call
                vector_search again with a reformulated query.
                Once you have good context, call classify_task and then
                summarize_context before writing your Final Answer.

                Query: {query}
            """).strip()
        }

    @staticmethod
    def _build_zero_shot_input(query: str) -> dict:
        return {
            "input": textwrap.dedent(f"""\
                No knowledge base is loaded. Answer the query using your general knowledge.
                You may still call classify_task and summarize_context, but do NOT call
                vector_search as no index is available.

                Query: {query}
            """).strip()
        }

    @staticmethod
    def _extract_sources(intermediate_steps: list) -> list[str]:
        """Pull content strings out of vector_search tool observations."""
        sources: list[str] = []
        for action, observation in intermediate_steps:
            if getattr(action, "tool", "") == "vector_search":
                try:
                    items = json.loads(observation)
                    for item in items:
                        if isinstance(item, dict) and "content" in item:
                            sources.append(item["content"][:200])
                except (json.JSONDecodeError, TypeError):
                    pass
        return sources
