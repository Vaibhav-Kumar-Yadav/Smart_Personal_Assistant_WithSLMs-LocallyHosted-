from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from config import CONFIG
from ingest_utils import load_documents, process_documents, save_vectors
from logger import logger

class DocumentIngester:
    def __init__(self, embeddings_model_name: str):
        self.model = SentenceTransformer(embeddings_model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    def ingest_documents(self, source_dir: str, vector_store_path: str) -> bool:
        try:
            documents = load_documents(source_dir)
            if not documents:
                logger.info("No new documents found to ingest.")
                return False

            processed_documents = process_documents(documents)
            logger.info(f"Ingested {len(processed_documents)} document chunks")

            vectors = np.array([
                self.model.encode(doc.page_content) 
                for doc in processed_documents
            ])
            metadata = [doc.metadata for doc in processed_documents]

            save_vectors(vectors, metadata, processed_documents, vector_store_path)
            logger.info(f"Saved vectors and metadata to {vector_store_path}.pkl")
            return True

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            return False

def main():
    ingester = DocumentIngester(CONFIG['embeddings_model_name'])
    success = ingester.ingest_documents(CONFIG['source_dir'], 'vectors.faiss')
    if not success:
        logger.error("Document ingestion failed")

if __name__ == "__main__":
    main()