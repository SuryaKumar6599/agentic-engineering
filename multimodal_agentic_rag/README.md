# Multimodal Agentic RAG

A completely free, local, and open-source **True Multimodal Agentic RAG** pipeline. 

This project uses LangChain, FAISS, PyTorch, and Ollama to build an intelligent RAG system capable of "seeing" images in PDFs and answering questions contextually.

## Features
- **Multimodal Ingestion**: Extracts text and images from PDFs. Uses `qwen2.5vl:7b` to generate rich visual descriptions of embedded charts/images.
- **Semantic Chunking**: Uses LangChain's `SemanticChunker` (backed by `nomic-embed-text`) to dynamically split text based on semantic meaning rather than arbitrary character counts.
- **Agentic Routing**: Uses `qwen2.5:7b` to evaluate retrieved FAISS context before answering, rejecting queries if the context is irrelevant.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*Note: Ensure Ollama is installed and running with `qwen2.5:7b`, `qwen2.5vl:7b`, and `nomic-embed-text`.*

## Usage
**1. Ingest a Document:**
```bash
python run_rag.py --ingest path/to/document.pdf
```
**2. Chat with the Agent:**
```bash
python run_rag.py --chat
```
