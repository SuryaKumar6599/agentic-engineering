import os
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "nomic-embed-text"
INDEX_PATH = "faiss_index"

def build_vectorstore(documents: List[Document]) -> FAISS:
    """Embeds documents into a FAISS index using Nomic."""
    logger.info(f"Embedding {len(documents)} chunks into FAISS...")
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    vectorstore = FAISS.from_documents(documents, embedder)
    
    # Save the index locally
    vectorstore.save_local(INDEX_PATH)
    logger.info(f"FAISS index saved to {INDEX_PATH}.")
    return vectorstore

def load_vectorstore() -> FAISS:
    """Loads an existing FAISS index."""
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Please ingest documents first.")
        
    logger.info(f"Loading FAISS index from {INDEX_PATH}...")
    return FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)
