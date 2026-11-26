from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from configs.settings import settings
from schemas.models import TaskCategory, TaskResult

_SYSTEM_PROMPT = """\
You are a task classification assistant. Given a user query and optional retrieved context,
classify the query into exactly one of the following categories:

{categories}

Return a structured response with:
- category: one of the categories above
- confidence: float between 0.0 and 1.0
- rationale: a single sentence explaining your classification decision

Be concise and decisive. If context is empty, classify based on the query alone.
"""

_HUMAN_PROMPT = """\
Query: {query}

Retrieved context (may be empty):
{context}
"""

_CATEGORIES = "\n".join(f"- {c.value}" for c in TaskCategory)


class TaskDetector:
    """Classify a user query into a TaskCategory using structured LLM output.

    Uses .with_structured_output() to guarantee valid Pydantic output via
    OpenAI function-calling — no JSON parsing, no try/except on format.
    """

    def __init__(self) -> None:
        base_llm = ChatOpenAI(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            temperature=0.0,
        )
        self._llm = base_llm.with_structured_output(TaskResult)

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])

        self._chain = self._prompt | self._llm

    def detect(self, query: str, context: str = "") -> TaskResult:
        """Return a validated TaskResult for the given query + context."""
        result = self._chain.invoke({
            "categories": _CATEGORIES,
            "query": query,
            "context": context or "(none)",
        })
        # with_structured_output raises ValidationError on bad output — no silent fallback.
        return result  # type: ignore[return-value]
