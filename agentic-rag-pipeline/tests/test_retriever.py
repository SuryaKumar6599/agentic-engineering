from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from pipeline.retriever import FAISSRetriever


class _MockEmbeddings:
    """Deterministic mock — returns fixed-length vectors without loading PyTorch."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384


@pytest.fixture
def mock_embeddings():
    return _MockEmbeddings()


class TestFAISSRetrieverEmptyIndex:
    """No index on disk → retrieve() must return [] not a string."""

    def test_retrieve_returns_empty_list_when_no_index(self, mock_embeddings, tmp_path):
        retriever = FAISSRetriever(
            index_path=str(tmp_path / "nonexistent"),
            embeddings=mock_embeddings,
        )
        result = retriever.retrieve("any query")
        assert result == [], "Expected empty list, not an error string"
        assert isinstance(result, list)

    def test_is_loaded_false_when_no_index(self, mock_embeddings, tmp_path):
        retriever = FAISSRetriever(
            index_path=str(tmp_path / "nonexistent"),
            embeddings=mock_embeddings,
        )
        assert retriever.is_loaded is False

    def test_save_raises_when_no_store(self, mock_embeddings, tmp_path):
        retriever = FAISSRetriever(
            index_path=str(tmp_path / "idx"),
            embeddings=mock_embeddings,
        )
        with pytest.raises(RuntimeError, match="No index to save"):
            retriever.save()


class TestFAISSRetrieverWithDocuments:
    def test_add_documents_creates_store(self, mock_embeddings, tmp_path):
        retriever = FAISSRetriever(
            index_path=str(tmp_path / "idx"),
            embeddings=mock_embeddings,
        )
        docs = [Document(page_content="hello world", metadata={"source": "test"})]

        # We patch FAISS.from_documents to avoid the real embedding call.
        fake_store = MagicMock()
        fake_store.max_marginal_relevance_search.return_value = docs

        with patch("pipeline.retriever.FAISS.from_documents", return_value=fake_store):
            retriever.add_documents(docs)

        assert retriever.is_loaded is True

    def test_retrieve_calls_mmr(self, mock_embeddings, tmp_path):
        retriever = FAISSRetriever(
            index_path=str(tmp_path / "idx"),
            embeddings=mock_embeddings,
        )
        fake_doc = Document(page_content="test chunk")
        fake_store = MagicMock()
        fake_store.max_marginal_relevance_search.return_value = [fake_doc]

        with patch("pipeline.retriever.FAISS.from_documents", return_value=fake_store):
            retriever.add_documents([fake_doc])

        results = retriever.retrieve("test query", k=3)
        fake_store.max_marginal_relevance_search.assert_called_once()
        call_kwargs = fake_store.max_marginal_relevance_search.call_args
        assert call_kwargs.kwargs.get("k") == 3 or call_kwargs.args[1] == 3
