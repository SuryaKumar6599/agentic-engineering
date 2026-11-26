from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pipeline.retriever import FAISSRetriever, get_embedding_model

# Supported extensions → loader class
_LOADERS: dict[str, type] = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
}

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)


class DocumentIngester:
    """Load, chunk, embed, and store documents into a FAISSRetriever.

    Parameters
    ----------
    retriever:
        Target retriever. If None, a new one is created (uses the singleton embeddings).
    """

    def __init__(self, retriever: Optional[FAISSRetriever] = None) -> None:
        self.retriever = retriever or FAISSRetriever(embeddings=get_embedding_model())

    def ingest_path(self, path: str) -> int:
        """Ingest a file or directory. Returns the total number of chunks stored."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        raw_docs: list[Document] = []
        if target.is_file():
            raw_docs = self._load_file(target)
        else:
            for fp in sorted(target.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in _LOADERS:
                    raw_docs.extend(self._load_file(fp))

        if not raw_docs:
            raise ValueError(f"No supported documents found at: {path}")

        chunks = _SPLITTER.split_documents(raw_docs)
        self.retriever.add_documents(chunks)
        self.retriever.save()
        return len(chunks)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> list[Document]:
        ext = path.suffix.lower()
        loader_cls = _LOADERS.get(ext)
        if loader_cls is None:
            return []
        try:
            loader = loader_cls(str(path))
            return loader.load()
        except Exception as exc:
            # Log and skip unreadable files rather than aborting the whole ingest.
            print(f"[warn] Could not load {path}: {exc}")
            return []
