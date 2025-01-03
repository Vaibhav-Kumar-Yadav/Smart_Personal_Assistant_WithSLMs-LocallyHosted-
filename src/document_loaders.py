from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
)

LOADER_CLASSES = {
    "CSVLoader": CSVLoader,
    "TextLoader": TextLoader,
    "UnstructuredEmailLoader": UnstructuredEmailLoader,
    "UnstructuredEPubLoader": UnstructuredEPubLoader,
    "UnstructuredHTMLLoader": UnstructuredHTMLLoader,
    "UnstructuredODTLoader": UnstructuredODTLoader,
    "UnstructuredPowerPointLoader": UnstructuredPowerPointLoader,
    "UnstructuredWordDocumentLoader": UnstructuredWordDocumentLoader,
    "PyPDFLoader": PyPDFLoader,
}

def get_loader(loader_name: str, file_path: str, loader_args: dict):
    loader_class = LOADER_CLASSES.get(loader_name)
    if not loader_class:
        raise ValueError(f"Unsupported loader: {loader_name}")
    return loader_class(file_path, **loader_args)