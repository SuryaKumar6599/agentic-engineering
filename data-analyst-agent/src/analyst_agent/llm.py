"""Pluggable LLM adapters (OpenAI, Anthropic, Gemini) and deterministic offline client."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


def _post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc


class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIClient.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        return data["choices"][0]["message"]["content"]


class AnthropicClient:
    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-latest") -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicClient.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            {
                "model": self.model,
                "max_tokens": 2048,
                "temperature": 0.0,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )
        return data["content"][0]["text"]


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiClient.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        data = _post_json(
            url,
            {},
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.0},
            },
        )
        return data["candidates"][0]["content"]["parts"][0]["text"]


class OllamaClient:
    """LLM client for any model served by a local Ollama instance."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise RuntimeError(
                f"Ollama connection failed at {self.base_url}. "
                f"Is 'ollama serve' running? Error: {exc}"
            ) from exc


class ScriptedLLMClient:
    """Deterministic scripted LLM client for testing and offline failure-mode demos."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise RuntimeError("ScriptedLLMClient ran out of scripted responses.")
        return self._responses.pop(0)


def auto_select_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(model=model or "qwen2.5:7b")
    if provider == "openai" or (provider is None and os.environ.get("OPENAI_API_KEY")):
        return OpenAIClient(model=model or "gpt-4o-mini")
    if provider == "anthropic" or (provider is None and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicClient(model=model or "claude-3-5-sonnet-latest")
    if provider == "gemini" or (provider is None and os.environ.get("GEMINI_API_KEY")):
        return GeminiClient(model=model or "gemini-2.5-flash")
    # Last resort: try Ollama on localhost
    raise RuntimeError(
        "No LLM configured. Options:\n"
        "  • Local Ollama:  --provider ollama  (ollama must be running)\n"
        "  • OpenAI:        set OPENAI_API_KEY\n"
        "  • Anthropic:     set ANTHROPIC_API_KEY\n"
        "  • Gemini:        set GEMINI_API_KEY"
    )
