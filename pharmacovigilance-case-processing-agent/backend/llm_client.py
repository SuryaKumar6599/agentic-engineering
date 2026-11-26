from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
FAST_MODEL = os.environ.get("OLLAMA_FAST_MODEL", "qwen2.5:7b")
REASONING_MODEL = os.environ.get("OLLAMA_REASONING_MODEL", "qwen3.6:27b")


def get_ollama_status() -> Dict[str, Any]:
    """Check local Ollama availability and installed models."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.5)
        resp.raise_for_status()
        models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
        return {
            "available": True,
            "base_url": OLLAMA_BASE_URL,
            "fast_model": FAST_MODEL if FAST_MODEL in models else (models[0] if models else FAST_MODEL),
            "reasoning_model": REASONING_MODEL if REASONING_MODEL in models else (FAST_MODEL if FAST_MODEL in models else (models[0] if models else REASONING_MODEL)),
            "installed_models": models,
        }
    except Exception as exc:
        return {
            "available": False,
            "base_url": OLLAMA_BASE_URL,
            "fast_model": FAST_MODEL,
            "reasoning_model": REASONING_MODEL,
            "installed_models": [],
            "error": str(exc),
        }


def ollama_chat_json(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    timeout: float = 90.0,
) -> Optional[Dict[str, Any]]:
    """Call local Ollama /api/chat with JSON format; returns parsed dict or None on failure."""
    target_model = model or FAST_MODEL
    try:
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 220},
        }
        resp = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        if not content:
            return None
        return json.loads(content)
    except Exception:
        return None


def ollama_chat_text(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    timeout: float = 60.0,
) -> Optional[str]:
    """Call local Ollama /api/chat for free-text synthesis (e.g., ICH clinical narrative)."""
    target_model = model or REASONING_MODEL
    try:
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        return content or None
    except Exception:
        return None
