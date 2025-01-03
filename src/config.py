import json

def load_config():
    try:
        with open('config.json') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("config.json not found")

CONFIG = load_config()

LOADER_MAPPING = {
    ".csv": ("CSVLoader", {}),
    ".doc": ("UnstructuredWordDocumentLoader", {}),
    ".docx": ("UnstructuredWordDocumentLoader", {}),
    ".eml": ("UnstructuredEmailLoader", {}),
    ".epub": ("UnstructuredEPubLoader", {}),
    ".html": ("UnstructuredHTMLLoader", {}),
    ".md": ("TextLoader", {}),
    ".odt": ("UnstructuredODTLoader", {}),
    ".pdf": ("PyPDFLoader", {}),
    ".ppt": ("UnstructuredPowerPointLoader", {}),
    ".pptx": ("UnstructuredPowerPointLoader", {}),
    ".txt": ("TextLoader", {"encoding": "utf8"}),
}

MODEL_CONFIG = {
    "search_options": {
        "max_length": 2048,
        "temperature": 0.5,
        "top_p": 0.5,
        "top_k": 40,
        "repetition_penalty": 1.1
    },
    "chat_template": {
        "system": """<|user|>
System: {system}
Context: {context}
Citations: {citations}
User Question: {input}
Remember to answer strictly based on the provided context. If the information is not in the context, say "I don't find this information in the provided context."
<|end|>
<|assistant|>"""
    }
}

SYSTEM_PROMPT = """You are Jarvis, an AI assistant that provides accurate information based strictly on the given context. 
Follow these rules:
1. Only use information from the provided context
2. If the information isn't in the context, say "I don't find this information in the provided context"
3. Cite sources using [1], [2], [3] etc. when referring to specific information
4. Keep responses clear and concise"""