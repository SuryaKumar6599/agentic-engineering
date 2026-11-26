import logging
from typing import List, Tuple
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from .vectorstore import load_vectorstore

logger = logging.getLogger(__name__)
TEXT_MODEL = "qwen2.5:7b"

# Prompt for the Agent to evaluate if the context is relevant before answering
EVAL_PROMPT = """
You are an expert evaluator. You must determine if the retrieved context contains the answer to the user's question.

Question: {question}

Retrieved Context:
{context}

If the context contains the answer (or enough information to formulate an answer), output exactly "YES".
If the context is irrelevant or does not contain the answer, output exactly "NO".
"""

# Prompt for the Agent to generate the final answer
QA_PROMPT = """
You are a highly intelligent, multimodal RAG agent. You have extracted both text and visual descriptions from a document.
Use the following pieces of retrieved context to answer the question. 
If the context mentions "IMAGE/PAGE X VISUAL SUMMARY", treat it as a description of an image on that page.
If you don't know the answer based on the context, just say that you don't know.

Question: {question}

Context:
{context}

Answer concisely and accurately:
"""

class AgenticRAG:
    def __init__(self):
        logger.info(f"Initializing Qwen Agent with {TEXT_MODEL}...")
        self.llm = Ollama(model=TEXT_MODEL, temperature=0.1)
        self.vectorstore = load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        self.eval_prompt = PromptTemplate(input_variables=["question", "context"], template=EVAL_PROMPT)
        self.qa_prompt = PromptTemplate(input_variables=["question", "context"], template=QA_PROMPT)

    def ask(self, question: str) -> str:
        """Agentic loop: Retrieve -> Evaluate -> Answer (or reject)."""
        logger.info(f"Agent received question: '{question}'")
        
        # 1. Retrieval
        docs = self.retriever.invoke(question)
        context = "\n\n".join([f"[{d.metadata.get('type', 'text')}] {d.page_content}" for d in docs])
        logger.info(f"Retrieved {len(docs)} chunks from the vector database.")
        
        # 2. Evaluation (Agentic routing)
        eval_chain = self.eval_prompt | self.llm
        decision = eval_chain.invoke({"question": question, "context": context}).strip().upper()
        
        if "NO" in decision:
            logger.warning("Agent evaluated the retrieved context as IRRELEVANT. Rejecting question.")
            return "I'm sorry, but I cannot find relevant information in the uploaded documents to answer this question."
            
        logger.info("Agent evaluated the context as RELEVANT. Generating answer...")
        
        # 3. Generation
        qa_chain = self.qa_prompt | self.llm
        answer = qa_chain.invoke({"question": question, "context": context})
        
        return answer
