import streamlit as st
from rag_backend import RAGBackend
from log import setup_logging, log_info, log_error
import shutil
import os

# Configure logging
logger = setup_logging()

# Define system prompt for RAG
SYSTEM_PROMPT = """You are a helpful AI assistant specializing in answering questions based on the provided knowledge base.

Guidelines:
- Answer questions accurately based only on the provided context
- If information is not in the context, clearly state that you don't have that information
- Provide clear, concise, and well-structured responses
- When appropriate, include code examples or step-by-step instructions
- Be friendly and professional in your responses"""

# Sidebar configuration
st.sidebar.header("🔧 Backend Configuration")

# Backend selection
backend_type = st.sidebar.selectbox(
    "Select Backend:",
    [ "ovms", "ollama"],
    help="Choose between Ollama or OpenVINO Model Server (OVMS)"
)

# Backend-specific configuration
if backend_type == "ollama":
    ollama_url = st.sidebar.text_input(
        "Ollama URL:", 
        value="http://localhost:11434",
        help="URL where Ollama server is running"
    )
    model_name = st.sidebar.text_input(
        "LLM Model:", 
        value="qwen2.5:7b-instruct",
        help="Ollama model name for text generation"
    )
    embed_model = st.sidebar.text_input(
        "Embedding Model:", 
        value="bge-large",
        help="Ollama model name for embeddings"
    )
    
    # Initialize RAG Backend for Ollama
    rag = RAGBackend(
        backend_type="ollama",
        ollama_url=ollama_url,
        model_name=model_name,
        embed_model=embed_model,
        system_prompt=SYSTEM_PROMPT
    )

else:  # OVMS
    ovms_url = st.sidebar.text_input(
        "OVMS URL:", 
        value="http://localhost:8000/v3",
        help="OpenAI-compatible OVMS endpoint URL"
    )
    model_name = st.sidebar.text_input(
        "LLM Model:", 
        value="Qwen/Qwen2-7B-Instruct",
        help="OVMS model name for text generation (matches TEXT_GENERATION_SOURCE_MODELS)"
    )
    embed_model = st.sidebar.text_input(
        "Embedding Model:", 
        value="BAAI/bge-base-en-v1.5",
        help="OVMS model name for embeddings (matches EMBEDDINGS_SOURCE_MODEL)"
    )
    
    # Initialize RAG Backend for OVMS
    rag = RAGBackend(
        backend_type="ovms",
        ovms_url=ovms_url,
        model_name=model_name,
        embed_model=embed_model,
        system_prompt=SYSTEM_PROMPT
    )
    
    # Show available OVMS models
    st.sidebar.info("""
    **Available OVMS Models:**
    - **Text Generation:** Qwen/Qwen2-7B-Instruct
    - **Embeddings:** BAAI/bge-base-en-v1.5  
    - **Rerank:** BAAI/bge-reranker-base
    """)
    
    st.sidebar.warning("""
    **Note:** Make sure your OVMS container is running:
    ```
    cd ovms && docker-compose up -d
    ```
    """)

# Cached wrapper for index loading
@st.cache_resource
def load_index():
    """Cached wrapper for index loading"""
    return rag.load_data_and_create_index()

# Streamlit UI
st.title("🤖 RAG Chat with LlamaIndex")
st.markdown("Ask questions about the knowledge base")

config = rag.get_config_info()
log_info("🚀 Starting RAG Chat application...")
log_info(f"⚙️ Configuration - Backend: {config['backend_type']}, LLM: {config['llm_model']}, Embedding: {config['embed_model']}")

# Check backend connectivity
if not rag.check_backend_connection():
    backend_url = config.get('ollama_url', config.get('ovms_url', 'unknown'))
    st.error(f"❌ Cannot connect to {config['backend_type'].upper()} at {backend_url}")
    if config['backend_type'] == 'ollama':
        st.info("Please make sure Ollama is running by executing: `ollama serve`")
    else:
        st.info("Please make sure OVMS container is running and accessible")
    st.stop()

# Check if models are available
llm_available, embed_available = rag.check_models_available()
if not llm_available:
    st.error(f"❌ LLM model '{config['llm_model']}' not found")
    if config['backend_type'] == 'ollama':
        st.info(f"Please pull the model: `ollama pull {config['llm_model']}`")
    else:
        st.info(f"Please ensure the model '{config['llm_model']}' is deployed in OVMS")
    st.stop()

if not embed_available:
    st.error(f"❌ Embedding model '{config['embed_model']}' not found")
    if config['backend_type'] == 'ollama':
        st.info(f"Please pull the model: `ollama pull {config['embed_model']}`")
    else:
        st.info(f"Please ensure the model '{config['embed_model']}' is deployed in OVMS")
    st.stop()

st.success(f"✅ {config['backend_type'].upper()} connection and models verified!")
log_info(f"✅ {config['backend_type'].upper()} connection and models verified successfully")

# Initialize the index
log_info("📚 Initializing knowledge base...")
with st.spinner("Loading knowledge base..."):
    index = load_index()

if index is None:
    st.error("❌ Failed to load knowledge base!")
    st.stop()

st.success("✅ Knowledge base loaded successfully!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the knowledge base..."):
    log_info(f"📝 User query: {prompt}")
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating response..."):
            response = rag.docs_retrieve(prompt, index)

            if response:
                # Display the response
                response_placeholder = st.empty()
                
                try:
                    # Stream the response using the timed streaming function
                    full_response = rag.stream_response(response, response_placeholder)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Show source documents in expander
                    with st.expander("📚 Source Documents"):
                        for i, node in enumerate(response.source_nodes):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"*Score: {node.score:.3f}*")
                            st.markdown(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                            st.markdown("---")
                    
                    log_info(f"✅ Query completed with {len(response.source_nodes)} source documents")
                
                except Exception as e:
                    log_error(f"❌ Error while streaming: {e}")
                    st.error(f"Error while streaming response: {e}")
                    # Fallback to non-streaming response
                    if hasattr(response, 'response'):
                        log_info("🔄 Falling back to non-streaming response")
                        response_placeholder.markdown(response.response)
                        st.session_state.messages.append({"role": "assistant", "content": response.response})
                    else:
                        st.error("Could not retrieve response. Please try again.")
            else:
                log_error("❌ Failed to get response")
                st.error("Failed to get response from the knowledge base.")

# Sidebar with information
with st.sidebar:
    st.markdown("---")
    st.header("� Current Configuration")
    st.code(f"Backend: {config['backend_type'].upper()}")
    st.code(f"LLM Model: {config['llm_model']}")
    st.code(f"Embedding Model: {config['embed_model']}")
    
    if config['backend_type'] == 'ollama':
        st.code(f"Ollama URL: {config['ollama_url']}")
    else:
        st.code(f"OVMS URL: {config['ovms_url']}")
    
    with st.expander("📝 System Prompt"):
        st.text_area("Current System Prompt:", value=config['system_prompt'], height=200, disabled=True)
    
    if st.button("🔄 Reload Knowledge Base"):
        log_info("Reloading knowledge base...")
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("🗑️ Rebuild Vector Database", type="secondary"):
        log_info("Rebuilding vector database from scratch...")
        with st.spinner("Deleting old index and rebuilding from data directory..."):
            try:
                # Delete existing index directory to force recreation
                index_dir = "index_storage"
                if os.path.exists(index_dir):
                    shutil.rmtree(index_dir)
                    log_info(f"Deleted existing index at {index_dir}")
                
                # Clear Streamlit cache
                st.cache_resource.clear()
                log_info("Cleared Streamlit cache")
                
                # Rebuild the index immediately
                log_info("Creating new vector database...")
                new_index = rag.load_data_and_create_index()
                
                if new_index:
                    st.success("✅ Vector database rebuilt successfully!")
                    st.info("The knowledge base has been recreated with fresh embeddings from the `data/` directory.")
                    log_info("✅ Vector database rebuilt successfully")
                    
                    # Force a rerun to refresh the cached index
                    st.rerun()
                else:
                    st.error("❌ Failed to rebuild vector database")
                    log_error("Failed to rebuild vector database")
                
            except Exception as e:
                log_error(f"Error rebuilding vector database: {e}")
                st.error(f"Error rebuilding vector database: {e}")
    
    
