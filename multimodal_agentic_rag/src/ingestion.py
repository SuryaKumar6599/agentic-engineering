import os
import base64
import requests
import logging
from io import BytesIO
from typing import List, Dict, Any
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for Ollama API
OLLAMA_API_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "qwen2.5vl:7b"
EMBEDDING_MODEL = "nomic-embed-text"

def _encode_image_to_base64(image) -> str:
    """Converts a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def summarize_image_with_qwen(base64_image: str) -> str:
    """Passes the image to Qwen-VL to generate a descriptive summary."""
    payload = {
        "model": VISION_MODEL,
        "prompt": "Describe this image in detail. Extract any relevant charts, data, text, or figures. Be concise but highly descriptive.",
        "images": [base64_image],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Failed to generate image summary: {e}")
        return ""

def process_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts text and images from a PDF.
    - Text is semantically chunked.
    - Images are described by Qwen-VL and treated as text chunks.
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    # 1. Extract Text
    logger.info("Extracting and chunking text...")
    loader = PyPDFLoader(pdf_path)
    text_docs = loader.load()
    
    # Initialize Semantic Chunker with Nomic embeddings
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    semantic_chunker = SemanticChunker(embedder)
    
    # We combine all pages into one long string to let semantic chunker find the best splits
    full_text = "\n\n".join([doc.page_content for doc in text_docs])
    chunked_docs = semantic_chunker.create_documents([full_text])
    logger.info(f"Generated {len(chunked_docs)} semantic text chunks.")
    
    # 2. Extract Images
    logger.info("Extracting images for multimodal processing...")
    try:
        images = convert_from_path(pdf_path)
        logger.info(f"Found {len(images)} pages to analyze visually.")
        
        for i, img in enumerate(images):
            # For efficiency, you might want to skip mostly blank pages, 
            # but for this demo, we analyze every page as an image.
            logger.info(f"Summarizing page {i+1} visually with {VISION_MODEL}...")
            b64_img = _encode_image_to_base64(img)
            summary = summarize_image_with_qwen(b64_img)
            
            if summary:
                # Add the image summary as a Document chunk
                doc = Document(
                    page_content=f"IMAGE/PAGE {i+1} VISUAL SUMMARY:\n{summary}",
                    metadata={"source": pdf_path, "type": "image_summary", "page": i+1}
                )
                chunked_docs.append(doc)
    except Exception as e:
        logger.error(f"Image extraction failed (make sure poppler is installed): {e}")
        
    return chunked_docs
