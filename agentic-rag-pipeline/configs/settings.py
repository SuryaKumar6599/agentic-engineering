from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration — loaded from .env, with typed defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider (Ollama local)
    ollama_api_key: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"

    # Embeddings (local HuggingFace)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # FAISS persistence path
    faiss_index_path: str = "data/faiss_index"

    # ReAct agent
    agent_max_iterations: int = 6
    agent_verbose: bool = True

    # MMR retrieval parameters
    retrieval_k: int = 5
    retrieval_fetch_k: int = 20
    retrieval_lambda_mult: float = 0.6


settings = Settings()
