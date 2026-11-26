from __future__ import annotations

import os
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from configs.settings import settings

# ---------------------------------------------------------------------------
# Singleton embedding model — loaded once per process lifetime.
# Tests can monkeypatch get_embedding_model() to inject a mock.
# ---------------------------------------------------------------------------
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the process-global HuggingFace embedding model, initialising it on first call."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


class FAISSRetriever:
    """FAISS-backed semantic retriever with MMR diversity filtering.

    Parameters
    ----------
    index_path:
        Directory for loading/saving the FAISS index. Defaults to settings value.
    embeddings:
        Embedding model instance. Defaults to the process singleton. Pass a mock
        during tests to avoid loading PyTorch weights.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
    ) -> None:
        self.index_path = index_path or settings.faiss_index_path
        self.embeddings = embeddings or get_embedding_model()
        self.store: Optional[FAISS] = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float | None = None,
    ) -> list[Document]:
        """Return up to *k* diverse documents via Maximal Marginal Relevance.

        Returns an empty list when no index is loaded — callers must handle
        this as the zero-shot branch rather than treating it as context.
        """
        if self.store is None:
            return []

        _k = k or settings.retrieval_k
        _fetch_k = fetch_k or settings.retrieval_fetch_k
        _lambda = lambda_mult if lambda_mult is not None else settings.retrieval_lambda_mult

        return self.store.max_marginal_relevance_search(
            query,
            k=_k,
            fetch_k=_fetch_k,
            lambda_mult=_lambda,
        )

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and add documents to the store, creating it if absent."""
        if self.store is None:
            self.store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.store.add_documents(docs)

    def save(self) -> None:
        """Persist the current index to disk."""
        if self.store is None:
            raise RuntimeError("No index to save — ingest documents first.")
        os.makedirs(self.index_path, exist_ok=True)
        self.store.save_local(self.index_path)

    @property
    def is_loaded(self) -> bool:
        return self.store is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_index(self) -> Optional[FAISS]:
        index_file = os.path.join(self.index_path, "index.faiss")
        if not os.path.exists(index_file):
            return None
        return FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
