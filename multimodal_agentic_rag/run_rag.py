import os
import argparse
import logging
from src.ingestion import process_pdf
from src.vectorstore import build_vectorstore
from src.agent import AgenticRAG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest(pdf_path: str):
    """Pipeline to ingest a document into the RAG system."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return
        
    print("="*50)
    print("1. INGESTION PHASE (Multimodal Semantic Chunking)")
    print("="*50)
    docs = process_pdf(pdf_path)
    
    print("\n" + "="*50)
    print("2. INDEXING PHASE (FAISS + Nomic Embeddings)")
    print("="*50)
    build_vectorstore(docs)
    print("\nIngestion complete! You can now ask questions using the --ask flag.")

def chat():
    """Interactive loop to talk to the Agent."""
    print("="*50)
    print("3. QA PHASE (Qwen Agentic RAG)")
    print("="*50)
    try:
        agent = AgenticRAG()
    except Exception as e:
        print(f"Failed to load RAG system: {e}. Did you ingest a document first?")
        return
        
    print("\nAgent ready! Type 'exit' to quit.\n")
    while True:
        question = input("\nUser> ")
        if question.lower() in ['exit', 'quit']:
            break
        
        print("\nAgent> Thinking...")
        answer = agent.ask(question)
        print(f"\nAgent> {answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Agentic RAG Pipeline")
    parser.add_argument("--ingest", type=str, help="Path to a PDF to ingest.")
    parser.add_argument("--chat", action="store_true", help="Start the interactive chat mode.")
    
    args = parser.parse_args()
    
    if args.ingest:
        ingest(args.ingest)
    elif args.chat:
        chat()
    else:
        parser.print_help()
