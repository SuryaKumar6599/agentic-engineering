"""
harness/config.py
─────────────────
Central settings loaded from environment variables / .env file.

Local model routing
───────────────────
Set EVAL_MODEL=local to route all LLM calls (agent + judge) through the
OpenAI-compatible endpoint of your local server:

  Ollama     → LOCAL_MODEL_BASE_URL=http://localhost:11434/v1
  LM Studio  → LOCAL_MODEL_BASE_URL=http://localhost:1234/v1
  llama.cpp  → LOCAL_MODEL_BASE_URL=http://localhost:8080/v1
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default local model name (Ollama tag for Qwen3-30B-A3B UD-Q2_K_XL)
_DEFAULT_LOCAL_MODEL = "qwen3:30b-a3b-ud-q2_k_xl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Cloud LLM keys (optional when using local model) ───────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # ── Local model (Ollama / LM Studio / llama.cpp) ───────────────────────────
    local_model_base_url: str = Field(
        default="http://localhost:11434/v1", alias="LOCAL_MODEL_BASE_URL"
    )
    local_model_api_key: str = Field(
        default="ollama", alias="LOCAL_MODEL_API_KEY"
    )
    local_model_name: str = Field(
        default=_DEFAULT_LOCAL_MODEL, alias="LOCAL_MODEL_NAME"
    )

    # ── Judge model (used by DeepEval / Ragas) ──────────────────────────────────
    # Set to "local" to use the local model above as the eval judge.
    eval_model: str = Field(default="local", alias="EVAL_MODEL")
    eval_max_concurrency: int = Field(default=2, alias="EVAL_MAX_CONCURRENCY")

    # ── Langfuse ────────────────────────────────────────────────────────────────
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )

    # ── Budget / latency gates ──────────────────────────────────────────────────
    cost_budget_usd: float = Field(default=9999.0, alias="COST_BUDGET_USD")
    latency_p95_max_s: float = Field(default=60.0, alias="LATENCY_P95_MAX_S")
    latency_p99_max_s: float = Field(default=120.0, alias="LATENCY_P99_MAX_S")

    # ── Derived helpers (read-only) ─────────────────────────────────────────────

    @property
    def is_local(self) -> bool:
        """True when all LLM calls should route to the local server."""
        return self.eval_model.lower() == "local"

    @property
    def resolved_model_name(self) -> str:
        """The actual model identifier to pass to the OpenAI client."""
        return self.local_model_name if self.is_local else self.eval_model

    @property
    def openai_client_kwargs(self) -> dict:
        """
        Keyword args for AsyncOpenAI / OpenAI constructor.
        Routes to local server when EVAL_MODEL=local.
        """
        if self.is_local:
            return {
                "base_url": self.local_model_base_url,
                "api_key": self.local_model_api_key,
            }
        return {"api_key": self.openai_api_key}


settings = Settings()
