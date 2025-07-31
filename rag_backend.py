import os
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from log import timing_decorator, log_info, log_error


class RAGBackend:
    """RAG Backend for handling all LlamaIndex operations"""
    
    def __init__(self, ollama_url="http://localhost:11434", 
                 model_name="qwen2.5:7b-instruct", 
                 embed_model="bge-large"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.embed_model = embed_model
        self._configure_settings()
    
    def _configure_settings(self):
        """Configure LlamaIndex settings"""
        Settings.llm = Ollama(
            model=self.model_name, 
            base_url=self.ollama_url,
            request_timeout=120.0  # 2 minutes timeout
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=self.embed_model, 
            base_url=self.ollama_url,
            request_timeout=60.0  # 1 minute timeout for embeddings
        )
    
    @timing_decorator
    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        log_info("Checking Ollama connection...")
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            is_connected = response.status_code == 200
            log_info(f"Ollama connection: {'✅ Connected' if is_connected else '❌ Failed'}")
            return is_connected
        except Exception as e:
            log_error(f"Ollama connection failed: {e}")
            return False

    @timing_decorator
    def check_models_available(self):
        """Check if required models are available"""
        log_info("Checking model availability...")
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                model_names = [model['name'] for model in models.get('models', [])]
                log_info(f"Available models: {model_names}")
                
                llm_available = any(self.model_name in name for name in model_names)
                embed_available = any(self.embed_model in name for name in model_names)
                
                log_info(f"LLM '{self.model_name}': {'✅ Available' if llm_available else '❌ Missing'}")
                log_info(f"Embedding '{self.embed_model}': {'✅ Available' if embed_available else '❌ Missing'}")
                
                return llm_available, embed_available
            return False, False
        except Exception as e:
            log_error(f"Model check failed: {e}")
            return False, False

    @timing_decorator
    def load_data_and_create_index(self, index_dir="index_storage", data_path="data"):
        """Load or create a persistent vector index"""
        try:
            log_info(f"Checking for persistent index at: {index_dir}")

            if os.path.exists(index_dir):
                log_info("Loading index from disk...")
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                index = load_index_from_storage(storage_context)
                log_info("Index loaded from disk.")
                return index
            else:
                log_info(f"Checking data directory: {data_path}")
                if not os.path.exists(data_path):
                    log_error(f"Data directory '{data_path}' not found!")
                    return None
                    
                log_info("Loading documents from directory...")
                documents = SimpleDirectoryReader(data_path).load_data()
                if not documents:
                    log_error("No documents found in data directory!")
                    return None
                    
                log_info(f"Loaded {len(documents)} documents")
                log_info("Creating vector index with embeddings...")
                index = VectorStoreIndex.from_documents(documents)
                log_info("Saving index to disk...")
                index.storage_context.persist(persist_dir=index_dir)
                log_info("Index saved to disk.")
                return index
        except Exception as e:
            log_error(f"Error loading or saving index: {e}")
            return None

    @timing_decorator
    def docs_retrieve(self, query, index):
        """Get response using RAG - retrieves docs from vector DB"""
        try:
            # Create query engine
            log_info("Creating query engine...")
            query_engine = index.as_query_engine(
                similarity_top_k=3,  # Retrieve top 3 most relevant chunks
                streaming=True
            )
            
            # Query the index
            log_info("Executing RAG query...")
            response = query_engine.query(query)
            log_info(f"Retrieved {len(response.source_nodes)} source documents")
            
            return response
        except Exception as e:
            log_error(f"Error getting RAG response: {e}")
            return None

    @timing_decorator
    def stream_response(self, response, response_placeholder):
        """Stream the LLM response and update UI in real-time"""
        try:
            log_info("🎬 Starting response streaming...")
            full_response = ""
            token_count = 0
            
            # Stream the response and update UI
            for token in response.response_gen:
                full_response += token
                token_count += 1
                response_placeholder.markdown(full_response + "▌")
            
            # Final response without cursor
            response_placeholder.markdown(full_response)
            
            log_info(f"Streaming completed: {token_count} tokens")
            return full_response
        except Exception as e:
            log_error(f"❌ Error while streaming: {e}")
            raise

    def get_config_info(self):
        """Get configuration information"""
        return {
            "llm_model": self.model_name,
            "embed_model": self.embed_model,
            "ollama_url": self.ollama_url
        }
