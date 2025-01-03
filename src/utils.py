import pickle
import os
from typing import List, Tuple, Any
from logger import logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CONFIG

def format_citations(contexts: List[Any]) -> Tuple[str, str]:
    formatted_contexts = []
    citations = []
    
    for i, doc in enumerate(contexts, 1):
        formatted_contexts.append(doc.page_content)
        citation = f"[{i}] "
        if 'source' in doc.metadata:
            citation += f"Source: {doc.metadata['source']}"
        if 'url' in doc.metadata:
            citation += f" (URL: {doc.metadata['url']})"
        if 'page' in doc.metadata:
            citation += f" - Page: {doc.metadata['page']}"
        citations.append(citation)
    
    return "\n\n".join(formatted_contexts), "\n".join(citations)

def load_or_ingest_documents(vector_store_path: str) -> list:
    try:
        if os.path.exists(f"{vector_store_path}.pkl"):
            with open(f"{vector_store_path}.pkl", "rb") as f:
                data = pickle.load(f)
            if 'documents' in data:
                return data['documents']
            logger.warning("No 'documents' key found in the vectors file.")
            return []
        logger.warning("Vectors file does not exist. Please ingest documents first.")
        return []
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def initialize_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name=CONFIG['embeddings_model_name'])
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        return None