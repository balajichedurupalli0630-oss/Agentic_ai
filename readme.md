# 🤖 RAG Agent - Intelligent Document Question Answering System

A production-ready **Retrieval-Augmented Generation (RAG)** system powered by **LangGraph** agents that enables intelligent conversations with your documents.


[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)


---

## ✨ Features

### Core Capabilities
- 📄 **Multi-Format Document Support** - PDF, TXT, CSV files
- 🧠 **Intelligent Agent** - LangGraph-powered conversational AI
- 💬 **Conversation Memory** - Maintains context across multiple turns
- 🔍 **Semantic Search** - Vector-based similarity search with ChromaDB
- ⚡ **Async Processing** - Fast, non-blocking operations
- 🎯 **Query Expansion** - Automatically generates alternative search queries
- 📊 **Streaming Responses** - Real-time response generation
- 🛠️ **Tool Calling** - Agent decides when to search documents
- 🔄 **Session Management** - Track conversations by session ID

### API Features
- RESTful endpoints for all operations
- Document upload and management
- Chat with streaming support
- Vector store statistics
- Health monitoring
- CORS enabled for web integration

---

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
│  (FastAPI)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  LangGraph      │
│  Agent          │◄────┐
└────┬────────────┘     │
     │                  │
     ▼                  │
┌─────────────┐    ┌────┴─────┐
│  RAG Tool   │───►│   LLM    │
└──────┬──────┘    │  (Groq)  │
       │           └──────────┘
       ▼
┌─────────────┐
│  Vector DB  │
│  (ChromaDB) │
└─────────────┘
```

### Agent Workflow
1. **User Input** → Agent receives message
2. **Decision** → Agent decides if document search is needed
3. **Tool Call** → If needed, calls RAG tool
4. **Retrieval** → Searches vector database
5. **Context** → Retrieves relevant document chunks
6. **Generation** → LLM generates answer with context
7. **Response** → Returns to user

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Agent Framework** | LangGraph |
| **LLM Provider** | Groq (GPT-OSS-20B), Ollama (Llama3) |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB |
| **API Framework** | FastAPI |
| **Document Loaders** | LangChain Community |
| **Text Splitting** | RecursiveCharacterTextSplitter |

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- pip or conda
- (Optional) Ollama for local LLM

### Step 1: Clone Repository
```bash
git clone https://github.com/balajichedurupalli0630-oss/Agentic_ai.git
cd Agentic_ai
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n  Agentic_ai python=3.10
conda activate Agentic_ai
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
langchain==0.1.0
langchain-groq==0.0.1
langchain-ollama==0.0.1
langchain-community==0.0.10
langgraph==0.0.20
chromadb==0.4.18
sentence-transformers==2.2.2
pymupdf==1.23.8
numpy==1.24.3
pydantic==2.5.0
python-multipart==0.0.6
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
VECTOR_STORE_PATH=./vector_store_v2  # Optional: custom path
```

**Get Groq API Key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up/Login
3. Generate API key
4. Copy to `.env` file

---

## ⚙️ Configuration

### Vector Store Settings
Edit `vector_base.py`:
```python
vectorstore = VectorStore(
    collection_name="pdf_documents",  # Change collection name
    persist_directory="./vector_store_v2"  # Change storage path
)
```

### Embedding Model
Edit `embeddings.py`:
```python
embedding_loader = Embeddings(
    model_name="all-MiniLM-L6-v2"  # Options:
    # "all-mpnet-base-v2" - Better quality, slower
    # "paraphrase-multilingual-MiniLM-L12-v2" - Multilingual
)
```

### LLM Configuration
Edit `llm.py`:
```python
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",  # Change model
        temperature=0,  # Adjust creativity (0-1)
    )
```

### Document Chunking
Edit `doc_load_chunk.py`:
```python
def split_documents(documents, 
                   chunk_size=800,      # Adjust chunk size
                   chunk_overlap=160):  # Adjust overlap
```

---

## 🚀 Usage

### 1. Index Your Documents

**Option A: Index entire folder**
```bash
# Place documents in ./data/ folder
python index_local_data.py
```

**Option B: Use API to upload**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

### 2. Start the Server
```bash
python main.py

# Or with custom settings
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server will start at: `http://localhost:8000`

### 3. Interact with the Agent

**Using Python:**
```python
import requests

# Chat with the agent
response = requests.post("http://localhost:8000/chat", json={
    "message": "What is this document about?",
    "session_id": "user123"
})

print(response.json()["response"])
```

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize the key points",
    "session_id": "user123"
  }'
```

**Using Web Interface:**
Navigate to `http://localhost:8000/docs` for interactive API documentation.

---

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /
GET /health
```
Check if the system is running.

**Response:**
```json
{
  "status": "online",
  "message": "System operational. 150 documents in vector store."
}
```

---

#### 2. Upload Document
```http
POST /upload
```
Upload and process a document.

**Request:**
- Form Data: `file` (PDF, TXT, or CSV)

**Response:**
```json
{
  "file_name": "document.pdf",
  "chunks_created": 45,
  "message": "Document uploaded and processed successfully",
  "total_documents": 150
}
```

---

#### 3. Chat (Non-Streaming)
```http
POST /chat
```
Send a message to the agent.

**Request:**
```json
{
  "message": "What are the main findings?",
  "session_id": "user123"
}
```

**Response:**
```json
{
  "response": "Based on the documents, the main findings are...",
  "session_id": "user123",
  "sources": null
}
```

---
`

**Response:**
```
data: {"content": "Based"}
data: {"content": " on"}
data: {"content": " the"}
...
```

---

#### 4. Vector Store Statistics
```http
GET /documents/stats
```

**Response:**
```json
{
  "total_documents": 150,
  "collection_name": "pdf_documents",
  "persist_directory": "./vector_store_v2"
}
```

---

#### 5. Clear All Documents
```http
DELETE /documents/clear
```

**Response:**
```json
{
  "message": "All documents cleared successfully"
}
```

---

#### 6. Delete Specific Document
```http
DELETE /documents/{doc_id}
```

**Response:**
```json
{
  "message": "Document doc_abc123 deleted successfully"
}
```

---

## 📂 Project Structure

```
Agentic_ai/
│
├── agent.py                 # LangGraph agent definition
├── main.py                  # FastAPI application
├── rag.py                   # RAG retrieval logic
├── tools.py                 # Agent tools (rag_answer)
├── state.py                 # Agent state schema
├── llm.py                   # LLM configuration
├── embeddings.py            # Embedding generation
├── vector_base.py           # ChromaDB vector store
├── doc_load_chunk.py        # Document loading & chunking
├── index_local_data.py      # Batch indexing script
│
├── data/                    # Place documents here
│   ├── document1.pdf
│   ├── document2.txt
│   └── ...
│
├── vector_store_v2/         # ChromaDB persistence
│   └── chroma.sqlite3
│
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🔍 How It Works

### 1. Document Ingestion
```
PDF/TXT/CSV → Load → Chunk (800 chars) → Embed → Store in ChromaDB
```

### 2. Query Processing
```
User Query → Agent Evaluates → Calls RAG Tool → Retrieves Docs → LLM Generates Answer
```

### 3. Agent Decision Making

The agent uses **tool calling** to decide when to search documents:

```python
# Agent sees user message
if agent_needs_information():
    tool_call = rag_answer(query="user's question")
    context = tool_returns_documents()
    answer = llm_generates_response(context)
else:
    answer = llm_responds_directly()
```

### 4. Retrieval Strategy

**Semantic Search:**
1. Convert query to embedding (384-dim vector)
2. Search ChromaDB using cosine similarity
3. Return top-k results above threshold (default: 0.3)

**Query Expansion (Optional):**
- Generate 3 alternative queries
- Search with each query
- Combine and deduplicate results

### 5. Memory Management

- **Conversation History**: Stored in LangGraph checkpointer
- **Session Tracking**: Each `session_id` maintains separate context
- **Summarization**: Automatically summarizes long conversations (>10 messages)

---

## 🚀 Advanced Features

### 1. Custom Retrieval Parameters

```python
response = requests.post("http://localhost:8000/chat", json={
    "message": "What is machine learning?",
    "session_id": "user123",
    "top_k": 10,              # Retrieve top 10 documents
    "score_threshold": 0.5,   # Higher threshold = stricter matching
    "use_expansion": True     # Enable query expansion
})
```

### 2. Multiple LLM Fallback

The system uses Groq by default, with Ollama as fallback:

```python
# Primary: Groq (fast, cloud-based)
llm = ChatGroq(model="openai/gpt-oss-20b")

# Fallback: Ollama (local, private)
llm_local = ChatOllama(model="llama3")
```

### 3. Document Metadata Filtering

Extend `rag.py` to filter by metadata:

```python
results = self.vector_store.collections.query(
    query_embeddings=[query_embedding],
    where={"source": "annual_report.pdf"},  # Filter by source
    n_results=top_k
)
```

### 4. Custom Agent System Prompt

Modify `STYLE_SYSTEM` in `agent.py`:

```python
STYLE_SYSTEM = """
You are a financial analyst assistant specializing in...
- Your custom instructions here
- Specific behaviors
- Output format preferences
"""
```

---

## 🐛 Troubleshooting

### Issue: "Model not found" error
**Solution:**
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull the model
ollama pull llama3
```

### Issue: ChromaDB persistence errors
**Solution:**
```bash
# Delete existing database
rm -rf vector_store_v2/

# Re-index documents
python index_local_data.py
```

### Issue: Out of memory during embedding
**Solution:**
```python
# In embeddings.py, reduce batch size
embeddings = self.model.encode(
    texts,
    batch_size=8,  # Reduce from 32
    ...
)
```

### Issue: Slow retrieval
**Solution:**
- Reduce `top_k` in retrieval
- Disable query expansion
- Use smaller embedding model
- Increase `score_threshold`

### Issue: Groq API rate limits
**Solution:**
```python
# Switch to Ollama
llm = Ollama_llm()  # In llm.py
```

---

## 🧪 Testing

### Run Manual Tests
```bash
# Test document loading
python -c "from doc_load_chunk import load_documents; print(load_documents('./data'))"

# Test embeddings
python -c "from embeddings import embedding_loader; print(embedding_loader.generate_embedding(['test']))"

# Test vector store
python -c "from vector_base import vectorstore; print(vectorstore.collections.count())"
```


### Health Check
```bash
curl http://localhost:8000/health
```

---

## 🎯 Performance Optimization

### Tips for Production:

1. **Caching**: Implement Redis for embedding cache
2. **Async**: Already implemented (FastAPI + async/await)
3. **Batching**: Process multiple queries in parallel
4. **Indexing**: Use HNSW parameters in ChromaDB
5. **Pruning**: Remove low-quality chunks periodically

**Example HNSW Configuration:**
```python
self.collections = self.client.get_or_create_collection(
    name=self.collection_name,
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,  # Higher = better quality
        "hnsw:M": 16                   # More connections
    }
)
```

---

## 📈 Roadmap

- [ ] Multi-modal support (images, tables)
- [ ] Advanced reranking (cross-encoders)
- [ ] Web search integration
- [ ] Fine-tuning on domain data
- [ ] Multi-agent collaboration
- [ ] Evaluation metrics dashboard
- [ ] Prompt optimization
- [ ] Database query tools

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---



## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Groq](https://groq.com/) - Fast LLM inference
- [SentenceTransformers](https://www.sbert.net/) - Embedding models

---

## 📧 Contact

For questions or support:
- Open an issue on GitHub
- Email: balajichedurupalli0630@gmail.com 


---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=balajichedurupalli0630-oss/rag-agent&type=Date)](https://star-history.com/#balajichedurupalli0630-oss/rag-agent&Date)

---

**Built with ❤️ for the AI community**
