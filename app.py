import streamlit as st
from rag_backend import RAGBackend
from log import setup_logging, log_info, log_error

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

# Initialize RAG Backend
rag = RAGBackend(
    ollama_url="http://localhost:11434",
    model_name="qwen2.5:7b-instruct", 
    embed_model="bge-large",
    system_prompt=SYSTEM_PROMPT
)

# Cached wrapper for index loading
@st.cache_resource
def load_index():
    """Cached wrapper for index loading"""
    return rag.load_data_and_create_index()

# Streamlit UI
st.title("🤖 RAG Chat with LlamaIndex")
st.markdown("Ask questions about the knowledge base (Python, ML, Streamlit)")

config = rag.get_config_info()
log_info("🚀 Starting RAG Chat application...")
log_info(f"⚙️ Configuration - LLM: {config['llm_model']}, Embedding: {config['embed_model']}, Ollama URL: {config['ollama_url']}")

# Check Ollama connectivity
if not rag.check_ollama_connection():
    st.error(f"❌ Cannot connect to Ollama at {config['ollama_url']}")
    st.info("Please make sure Ollama is running by executing: `ollama serve`")
    st.stop()

# Check if models are available
llm_available, embed_available = rag.check_models_available()
if not llm_available:
    st.error(f"❌ LLM model '{config['llm_model']}' not found")
    st.info(f"Please pull the model: `ollama pull {config['llm_model']}`")
    st.stop()

if not embed_available:
    st.error(f"❌ Embedding model '{config['embed_model']}' not found")
    st.info(f"Please pull the model: `ollama pull {config['embed_model']}`")
    st.stop()

st.success("✅ Ollama connection and models verified!")
log_info("✅ Ollama connection and models verified successfully")

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
    
    st.header("🔧 Configuration")
    st.code(f"LLM Model: {config['llm_model']}")
    st.code(f"Embedding Model: {config['embed_model']}")
    st.code(f"Ollama URL: {config['ollama_url']}")
    
    with st.expander("📝 System Prompt"):
        st.text_area("Current System Prompt:", value=config['system_prompt'], height=200, disabled=True)
    
    if st.button("🔄 Reload Knowledge Base"):
        log_info("Reloading knowledge base...")
        st.cache_resource.clear()
        st.rerun()
    
    
