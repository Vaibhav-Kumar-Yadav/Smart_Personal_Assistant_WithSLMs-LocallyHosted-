import glob
import os
import pickle
import numpy as np
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_loaders import get_loader
from config import LOADER_MAPPING
from logger import logger

def load_single_document(file_path: str) -> List:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_name, loader_args = LOADER_MAPPING[ext]
        loader = get_loader(loader_name, file_path, loader_args)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str) -> List:
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True))
    
    documents = []
    for file_path in all_files:
        try:
            documents.extend(load_single_document(file_path))
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    return documents

def process_documents(documents: List, chunk_size: int = 512, chunk_overlap: int = 32) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def save_vectors(vectors: np.ndarray, metadata: List[Dict], documents: List, vector_store_path: str):
    data_to_save = {
        "vectors": vectors,
        "metadata": metadata,
        "documents": documents
    }
    with open(f"{vector_store_path}.pkl", "wb") as f:
        pickle.dump(data_to_save, f)