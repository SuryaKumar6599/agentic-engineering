from __future__ import annotations

import json
from typing import Any

from langchain.tools import tool

from pipeline.retriever import FAISSRetriever, get_embedding_model
from pipeline.task_detector import TaskDetector

# ---------------------------------------------------------------------------
# Module-level singletons — initialised once per process.
# The retriever reuses the global embedding singleton.
# ---------------------------------------------------------------------------
_retriever: FAISSRetriever | None = None
_detector: TaskDetector | None = None


def _get_retriever() -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever(embeddings=get_embedding_model())
    return _retriever


def _get_detector() -> TaskDetector:
    global _detector
    if _detector is None:
        _detector = TaskDetector()
    return _detector


# ---------------------------------------------------------------------------
# @tool — LangChain tool definitions exposed to the ReAct agent
# ---------------------------------------------------------------------------


@tool
def vector_search(query: str) -> str:
    """Search the FAISS knowledge base for chunks relevant to *query*.

    Returns a JSON array of {content, source} objects, or an empty array if no
    index is loaded. The agent should call this tool first to gather context.
    """
    docs = _get_retriever().retrieve(query)
    if not docs:
        return json.dumps([])

    results = [
        {
            "content": doc.page_content,
            "source": f"[Local] {doc.metadata.get('source', 'unknown')}",
        }
        for doc in docs
    ]
    return json.dumps(results, ensure_ascii=False)


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information when the local knowledge base lacks sufficient context.
    
    Returns a JSON array of {content, source} objects from web results.
    """
    from duckduckgo_search import DDGS
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return json.dumps([])
        
        formatted_results = [
            {
                "content": res.get("body", ""),
                "source": f"[Web] {res.get('href', 'web')}"
            }
            for res in results
        ]
        return json.dumps(formatted_results, ensure_ascii=False)
    except Exception as e:
        return json.dumps([{"content": f"Web search failed: {str(e)}", "source": "error"}])


@tool
def classify_task(query: str, context: str = "") -> str:
    """Classify *query* into a task category using the retrieved *context*.

    Returns a JSON object with keys: category, confidence, rationale.
    Call this after vector_search has returned sufficient context.
    """
    result = _get_detector().detect(query=query, context=context)
    return json.dumps({
        "category": result.category.value,
        "confidence": result.confidence,
        "rationale": result.rationale,
    })


@tool
def summarize_context(input_text: str) -> str:
    """Produce a concise synthesis of the retrieved context and task category.

    Pass both the context and the detected task category as a single string.
    Returns a plain-text summary suitable for inclusion in the final answer.
    Call this as the last step before generating the Final Answer.
    """
    # This tool is intentionally lightweight — it re-uses the same LLM via a
    # direct chain so the agent's synthesis step stays within the ReAct loop.
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from configs.settings import settings

    llm = ChatOpenAI(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
        temperature=0.3,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert synthesis assistant. Your goal is to provide the final answer to the user's query.\n"
            "You will be given the Task Category, Workflow Plan (if any), and Retrieved Context.\n\n"
            "INSTRUCTIONS BASED ON CATEGORY:\n"
            "1. If Category is CODE_GENERATION or PLANNING: You MUST generate the requested code or plan. "
            "Use the Workflow Plan as a guide. You may use your own world knowledge if the context is insufficient. DO NOT refuse to answer.\n"
            "2. If Category is RESEARCH or GENERAL_QA: You MUST rely STRICTLY on the Retrieved Context. "
            "CRITICAL: If the provided context does not contain information relevant to the query, you MUST state: "
            "'I do not have enough information to answer this query based on the provided documents.' Do NOT hallucinate."
        )),
        ("human", "{input_text}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"input_text": input_text})
    return result.content  # type: ignore[return-value]


# Expose the retriever accessor so agent.py can set a custom one during tests.
def set_retriever(r: FAISSRetriever) -> None:
    global _retriever
    _retriever = r
