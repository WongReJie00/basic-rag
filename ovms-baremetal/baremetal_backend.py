
import os
import requests,time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import List, Any
from pydantic import Field
import json
from log import timing_decorator, log_info, log_error

class OVMSLLM(LLM):
    model_name: str = Field(description="The name of the OVMS model")
    api_base: str = Field(description="The base URL for the OVMS API")
    system_prompt: str = Field(default="", description="System prompt for the model")

    def __init__(self, model_name: str, api_base: str, system_prompt: str = "", **kwargs):
        super().__init__(model_name=model_name, api_base=api_base, system_prompt=system_prompt, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=1024,
            model_name=self.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            headers = {"Content-Type": "application/json"}
            if self.system_prompt:
                formatted_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = prompt
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": formatted_prompt}],
                "stream": False,
                "temperature": 0.1
            }
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                return CompletionResponse(text=text)
            else:
                log_error(f"OVMS LLM request failed: {response.status_code} {response.text}")
                raise Exception(f"OVMS LLM request failed: {response.status_code}")
        except Exception as e:
            log_error(f"Error getting completion from OVMS: {e}")
            raise

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            headers = {"Content-Type": "application/json"}
            if self.system_prompt:
                formatted_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = prompt
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": formatted_prompt}],
                "stream": True,
                "temperature": 0.1
            }
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=120,
                stream=True
            )
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        token = delta['content']
                                        yield CompletionResponse(text=token, delta=token)
                            except json.JSONDecodeError:
                                continue
            else:
                log_error(f"OVMS streaming request failed: {response.status_code} {response.text}")
                raise Exception(f"OVMS streaming request failed: {response.status_code}")
        except Exception as e:
            log_error(f"Error streaming from OVMS: {e}")
            raise

    def chat(self, messages, **kwargs): raise NotImplementedError("Use complete() instead")
    def stream_chat(self, messages, **kwargs): raise NotImplementedError("Use stream_complete() instead")
    async def acomplete(self, prompt, **kwargs): return self.complete(prompt, **kwargs)
    async def astream_complete(self, prompt, **kwargs): 
        for response in self.stream_complete(prompt, **kwargs): yield response
    async def achat(self, messages, **kwargs): raise NotImplementedError("Use acomplete() instead")
    async def astream_chat(self, messages, **kwargs): raise NotImplementedError("Use astream_complete() instead")

class OVMSEmbedding(BaseEmbedding):
    model_name: str = Field(description="The name of the OVMS model")
    api_base: str = Field(description="The base URL for the OVMS API")

    def __init__(self, model_name: str, api_base: str, **kwargs):
        super().__init__(model_name=model_name, api_base=api_base, **kwargs)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            headers = {"Content-Type": "application/json"}
            data = {"model": self.model_name, "input": texts}
            response = requests.post(f"{self.api_base}/embeddings", headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return [item["embedding"] for item in result["data"]]
            else:
                log_error(f"OVMS embedding failed: {response.status_code} {response.text}")
                raise Exception(f"OVMS embedding failed: {response.status_code}")
        except Exception as e:
            log_error(f"Error getting embeddings from OVMS: {e}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embeddings([query])[0]
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


# OVMS-only RAG backend logic (adapted from rag_backend.py)
class BaremetalRAGBackend:
    def __init__(self, ovms_url="http://localhost:8002/v3", model_name="Qwen/Qwen2-7B-Instruct", embed_model="BAAI/bge-base-en-v1.5", system_prompt=None):
        self.ovms_url = ovms_url
        self.model_name = model_name
        self.embed_model = embed_model
        self.system_prompt = system_prompt
        self._configure_settings()

    def _configure_settings(self):
        log_info("🔧 Configuring OVMS backend...")
        Settings.llm = OVMSLLM(
            model_name=self.model_name,
            api_base=self.ovms_url,
            system_prompt=self.system_prompt
        )
        Settings.embed_model = OVMSEmbedding(
            model_name=self.embed_model,
            api_base=self.ovms_url
        )

    @timing_decorator
    def check_backend_connection(self):
        """Check if OVMS is running and accessible"""
        log_info("Checking OVMS connection...")
        try:
            response = requests.get(f"http://localhost:8002/v1/config", timeout=5)
            is_connected = response.status_code == 200
            log_info(f"OVMS connection: {'✅ Connected' if is_connected else '❌ Failed'}")
            return is_connected
        except Exception as e:
            log_error(f"OVMS connection failed: {e}")
            return False

    @timing_decorator
    def check_models_available(self):
        """Check if required models are available in OVMS"""
        log_info("Checking OVMS model availability...")
        try:
            response = requests.get(f"http://localhost:8002/v1/config", timeout=5)
            if response.status_code == 200:
                config_data = response.json()
                model_names = list(config_data.keys())
                log_info(f"Available OVMS models: {model_names}")
                llm_available = self.model_name in model_names
                embed_available = self.embed_model in model_names
                log_info(f"LLM '{self.model_name}': {'✅ Available' if llm_available else '❌ Missing'}")
                log_info(f"Embedding '{self.embed_model}': {'✅ Available' if embed_available else '❌ Missing'}")
                return llm_available, embed_available
            return False, False
        except Exception as e:
            log_error(f"OVMS model check failed: {e}")
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
            log_info("Creating query engine...")
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                streaming=True
            )
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
            for token in response.response_gen:
                full_response += token
                token_count += 1
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            log_info(f"Streaming completed: {token_count} tokens")
            return full_response
        except Exception as e:
            log_error(f"❌ Error while streaming: {e}")
            raise

    def get_config_info(self):
        """Get configuration information"""
        config = {
            "backend_type": "ovms-baremetal",
            "llm_model": self.model_name,
            "embed_model": self.embed_model,
            "ovms_url": self.ovms_url,
            "system_prompt": self.system_prompt
        }
        return config
