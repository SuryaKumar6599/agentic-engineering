"""
agents/example_agent/agent.py
──────────────────────────────
A minimal tool-using agent that the eval harness runs by default.

Replace this with your real agent implementation.
The contract: accept a str input, return an AgentTrace.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from harness.cost import cost_usd
from harness.models import AgentTrace, ToolCall

# ── Tool definitions ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
]


# ── Fake tool executor (replace with real implementations) ─────────────────────

def _execute_tool(name: str, args: dict[str, Any]) -> str:
    if name == "search_web":
        return f"[MOCK] Top results for '{args['query']}': ..."
    if name == "calculator":
        try:
            # Safe eval for simple arithmetic only
            result = eval(args["expression"], {"__builtins__": {}})  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    if name == "get_weather":
        return f"[MOCK] Weather in {args['location']}: 22°C, Sunny"
    return f"[MOCK] Unknown tool: {name}"


# ── Main agent entrypoint ──────────────────────────────────────────────────────

async def run(task_input: str) -> AgentTrace:
    """
    Run the example agent on a single task input.
    Returns an AgentTrace with full provenance.

    When settings.is_local is True the client is pointed at the local
    OpenAI-compatible server (Ollama / LM Studio / llama.cpp) and
    settings.local_model_name is used instead of a cloud model.
    """
    from harness.config import settings

    client = AsyncOpenAI(**settings.openai_client_kwargs)
    model = settings.resolved_model_name
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": task_input},
    ]
    tool_calls_log: list[ToolCall] = []
    total_prompt = 0
    total_completion = 0
    start = time.perf_counter()

    # ── Agentic loop ───────────────────────────────────────────────────────────
    for _ in range(5):  # max 5 turns
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        usage = response.usage
        total_prompt += usage.prompt_tokens
        total_completion += usage.completion_tokens
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            break  # final answer

        # Execute each tool call
        for tc in msg.tool_calls:
            import json
            args = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, args)
            tool_calls_log.append(
                ToolCall(name=tc.function.name, arguments=args, result=result)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    final_output = msg.content or ""
    elapsed_ms = (time.perf_counter() - start) * 1_000

    return AgentTrace(
        trace_id=str(uuid.uuid4()),
        task_id="",  # filled in by runner
        input=task_input,
        output=final_output,
        tool_calls=tool_calls_log,
        latency_ms=elapsed_ms,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        model=model,
        cost_usd=cost_usd(model, total_prompt, total_completion),
    )
