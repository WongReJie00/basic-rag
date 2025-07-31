# basic-rag

A RAG (Retrieval-Augmented Generation) chat application using LlamaIndex, Ollama, and Streamlit.


## Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
3. Required models pulled in Ollama

## Quick Start for Ollama

### 1. Install Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve
```

### 2. Pull Required Models

```bash
# Pull the LLM model
ollama pull qwen2.5:7b-instruct

# Pull the embedding model
ollama pull bge-large
```


## Quick Start for OVMS
Refer to README.md in the `ovms` directory for detailed instructions on setting up OVMS to host models.


### 3. Setup Python Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd basic-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Add Your Documents

Place your documents (PDF, TXT, etc.) in the `data/` folder:

```
data/
├── python_best_practices.txt
├── machine_learning_basics.txt
└── your_documents.pdf
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Configuration

Default settings in `app.py`:

- **Ollama URL**: `http://localhost:11434`
- **LLM Model**: `qwen2.5:7b-instruct`
- **Embedding Model**: `bge-large`

## File Structure

```
basic-rag/
├── app.py              # Streamlit UI
├── rag_backend.py      # RAG backend logic
├── log.py              # Logging utilities
├── requirements.txt    # Python dependencies
├── data/              # Your documents
└── index_storage/     # Persistent vector index (auto-created)
```

## Usage

1. Start the app and wait for the knowledge base to load
2. Ask questions about your documents in the chat interface
3. View source documents used for each response

## Troubleshooting

- **Connection errors**: Ensure Ollama is running (`ollama serve`)
- **Model not found**: Pull required models with `ollama pull`
- **No documents**: Add files to the `data/` folder
- **Slow responses**: Normal for first-time model loading

## Architecture

- **Frontend**: Streamlit for web interface
- **Backend**: LlamaIndex for RAG pipeline
- **LLM**: Ollama for local inference
- **Storage**: Persistent vector index on disk
