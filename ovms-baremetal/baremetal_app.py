

import streamlit as st
from baremetal_backend import BaremetalRAGBackend
from log import setup_logging, log_info, log_error
from ovms_manager import OVMSManager
import shutil
import os
import atexit

# Configure logging
logger = setup_logging()

# Define system prompt for RAG
SYSTEM_PROMPT = """You are a helpful AI assistant specializing in answering questions based on the provided knowledge base.\n\nGuidelines:\n- Answer questions accurately based only on the provided context\n- If information is not in the context, clearly state that you don't have that information\n- Provide clear, concise, and well-structured responses\n- When appropriate, include code examples or step-by-step instructions\n- Be friendly and professional in your responses"""

# Start OVMS only once per Streamlit session, stop it when app exits
@st.cache_resource
def get_ovms_manager():
    ovms_bin_dir = os.path.join(os.getcwd(), "ovms", "bin")
    ovms_lib_dir = os.path.join(os.getcwd(), "ovms", "lib")
    config_path = os.path.join(os.getcwd(), "models", "config_rag.json")
    manager = OVMSManager(ovms_bin_dir, ovms_lib_dir, config_path, port=8002)
    if not manager.is_running():
        manager.start()
    atexit.register(manager.stop)
    return manager

manager = get_ovms_manager()

# Sidebar configuration
st.sidebar.header("🔧 Backend Configuration")
backend_type = "ovms"  # Only OVMS supported in this app

# Backend-specific configuration
ovms_url = st.sidebar.text_input(
    "OVMS URL:", value="http://localhost:8002/v3",
    help="OpenVINO Model Server endpoint URL (baremetal)"
)
model_name = st.sidebar.text_input(
    "LLM Model:", value="Qwen/Qwen2-7B-Instruct",
    help="OVMS model name for text generation"
)
embed_model = st.sidebar.text_input(
    "Embedding Model:", value="BAAI/bge-base-en-v1.5",
    help="OVMS model name for embeddings"
)

# Initialize RAG Backend for OVMS
rag = BaremetalRAGBackend(
    ovms_url=ovms_url,
    model_name=model_name,
    embed_model=embed_model,
    system_prompt=SYSTEM_PROMPT
)

# Cached wrapper for index loading
@st.cache_resource
def load_index():
    return rag.load_data_and_create_index()

# Streamlit UI
st.title("🤖 RAG Chat with OVMS Baremetal")
st.markdown("Ask questions about the knowledge base")

config = rag.get_config_info()
log_info("🚀 Starting RAG Chat application...")
log_info(f"⚙️ Configuration - Backend: {config['backend_type']}, LLM: {config['llm_model']}, Embedding: {config['embed_model']}")

# Check backend connectivity
if not rag.check_backend_connection():
    st.error(f"❌ Cannot connect to OVMS at {config['ovms_url']}")
    st.info("Please make sure OVMS is running on baremetal and accessible.")
    st.stop()

# Check if models are available
llm_available, embed_available = rag.check_models_available()
if not llm_available:
    st.error(f"❌ LLM model '{config['llm_model']}' not found")
    st.info(f"Please ensure the model '{config['llm_model']}' is deployed in OVMS")
    st.stop()

if not embed_available:
    st.error(f"❌ Embedding model '{config['embed_model']}' not found")
    st.info(f"Please ensure the model '{config['embed_model']}' is deployed in OVMS")
    st.stop()

st.success(f"✅ OVMS connection and models verified!")
log_info(f"✅ OVMS connection and models verified successfully")

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating response..."):
            response = rag.docs_retrieve(prompt, index)
            if response:
                response_placeholder = st.empty()
                try:
                    full_response = rag.stream_response(response, response_placeholder)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    with st.expander("📚 Source Documents"):
                        for i, node in enumerate(getattr(response, "source_nodes", [])):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"*Score: {getattr(node, 'score', 0):.3f}*")
                            st.markdown(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                            st.markdown("---")
                    log_info(f"✅ Query completed with {len(getattr(response, 'source_nodes', []))} source documents")
                except Exception as e:
                    log_error(f"❌ Error while streaming: {e}")
                    st.error(f"Error while streaming response: {e}")
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
    st.header("🖥️ Current Configuration")
    st.code(f"Backend: OVMS Baremetal")
    st.code(f"LLM Model: {config['llm_model']}")
    st.code(f"Embedding Model: {config['embed_model']}")
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
                index_dir = "index_storage"
                if os.path.exists(index_dir):
                    shutil.rmtree(index_dir)
                    log_info(f"Deleted existing index at {index_dir}")
                st.cache_resource.clear()
                log_info("Cleared Streamlit cache")
                log_info("Creating new vector database...")
                new_index = rag.load_data_and_create_index()
                if new_index:
                    st.success("✅ Vector database rebuilt successfully!")
                    st.info("The knowledge base has been recreated with fresh embeddings from the `data/` directory.")
                    log_info("✅ Vector database rebuilt successfully")
                    st.rerun()
                else:
                    st.error("❌ Failed to rebuild vector database")
                    log_error("Failed to rebuild vector database")
            except Exception as e:
                log_error(f"Error rebuilding vector database: {e}")
                st.error(f"Error rebuilding vector database: {e}")
