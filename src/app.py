import streamlit as st
from model_handler import Phi3Handler
from utils import format_citations, load_or_ingest_documents, initialize_embeddings
from config import CONFIG, SYSTEM_PROMPT
from langchain_community.vectorstores import FAISS
from logger import logger

def manage_session_state():
    """Initialize and manage session state variables."""
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

def initialize_model():
    """Initialize the Phi-3 model."""
    try:
        model_handler = Phi3Handler(CONFIG['model_name'])
        if model_handler.initialize():
            st.session_state.model_handler = model_handler
            st.session_state.model_loaded = True
            st.sidebar.success("Phi-3 model loaded successfully!")
            return True
        else:
            st.sidebar.error("Failed to load Phi-3 model")
            return False
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        st.sidebar.error(f"Error initializing model: {str(e)}")
        return False

def setup_page():
    """Configure the Streamlit page settings and display header."""
    st.set_page_config(
        page_title="Smart AI Personal Assistant",
        page_icon="🔎",
        layout="wide"
    )
    st.title("🔎 Smart AI Personal Assistant")
    
    description = """The aim of this project is to design and develop a smart personal assistant for employees using advanced AI technologies, particularly generative AI (Gen AI) and AI agents. 
    This assistant aims to overcome the current limitations of existing personal AI assistants by providing efficient, secure, and context-aware assistance for various personal and domain-specific tasks. 
    The primary objective is to enhance productivity and streamline task management through a personalized user experience."""
    st.write(description)

def display_chat_history():
    """Display the chat history with citations."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message:
                with st.expander("📚 Sources & Citations"):
                    st.markdown(message["citations"])

def display_context_and_citations(contexts):
    """Display context snippets and citations in an expander."""
    context_text, citations = format_citations(contexts)
    
    with st.expander("📚 Sources & Citations"):
        st.markdown("### References:")
        st.markdown(citations)
        
        st.markdown("### Relevant Context Snippets:")
        for i, doc in enumerate(contexts, 1):
            st.markdown(f"**Snippet {i}:**")
            st.markdown(doc.page_content)
            st.markdown("---")
    
    return context_text, citations

def handle_user_input(docsearch):
    """Process user input and generate response."""
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if st.session_state.model_loaded:
                try:
                    # Get relevant context
                    contexts = docsearch.similarity_search(prompt)[:3]
                    context_text, citations = display_context_and_citations(contexts)

                    # Generate response
                    generator = st.session_state.model_handler.generate_response(
                        SYSTEM_PROMPT, 
                        prompt, 
                        context_text,
                        citations
                    )
                    
                    full_response = ""
                    for response in generator:
                        full_response = response
                        formatted_response = full_response + "\n\n---\n*For detailed sources, click 'Sources & Citations' above.*"
                        message_placeholder.markdown(formatted_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_response,
                        "citations": citations
                    })
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    message_placeholder.error(f"Error processing your request: {str(e)}")
            else:
                message_placeholder.error("Please wait for the model to load.")

def setup_sidebar():
    """Setup sidebar controls."""
    with st.sidebar:
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Reload Model"):
            if st.session_state.model_handler:
                st.session_state.model_handler.cleanup()
            st.session_state.model_handler = None
            st.session_state.model_loaded = False
            st.rerun()

def main():
    """Main application function."""
    setup_page()
    manage_session_state()

    # Initialize model if not already loaded
    if st.session_state.model_handler is None:
        with st.spinner("Loading Phi-3 model..."):
            if not initialize_model():
                st.error("Failed to initialize the model. Please check the logs.")
                return

    # Initialize embeddings and load documents
    embeddings = initialize_embeddings()
    if not embeddings:
        st.error("Failed to initialize embeddings model")
        return

    try:
        documents = load_or_ingest_documents('vectors.faiss')
        if not documents:
            st.error("No documents found. Please make sure to ingest documents first.")
            return
        
        docsearch = FAISS.from_documents(documents, embeddings)
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        st.error("Error loading document database. Please check the logs.")
        return

    # Display chat interface
    display_chat_history()
    handle_user_input(docsearch)
    setup_sidebar()

if __name__ == "__main__":
    main()