# 🤖 Agentic RAG Assistant - Intelligent, Table-Aware Document Question Answering System

A production-grade **Table-Aware Retrieval-Augmented Generation (RAG)** system powered by **LangGraph** agents, **Hybrid Search (Dense Vector + Sparse BM25)**, **Cross-Encoder Reranking**, and **`pymupdf4llm` Markdown table parsing** with a modern institutional academic interface.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70b-purple.svg)](https://groq.com/)
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
- [Testing & Observability](#-testing--observability)
- [Performance Optimization](#-performance-optimization)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 📊 Table-Aware Ingestion & Chunking
- 📄 **`pymupdf4llm` PDF Parsing**: Extracts PDFs into GFM-style Markdown blocks (`| col | col |`).
- 🧩 **Atomic Table Blocks**: Heuristically detects tables (`is_table_block`) and preserves every table as a single, indivisible chunk — preventing mid-row or mid-cell splitting.
- 🏷️ **Metadata Tagging & Dedup**: Tags chunks with `content_type: "table"` or `"text"`, preserves original user filenames, and guarantees batch-level unique MD5 chunk keys (`doc_{src}_{i}_{content}`) to avoid duplicate errors.

### 🧠 Intelligent Hybrid Retrieval & Reranking
- 🔍 **Hybrid Retrieval**: Merges dense semantic vector search (ChromaDB) with sparse keyword search (`rank_bm25`).
- 🎯 **Cross-Encoder Reranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank top candidates for accurate context selection.
- 🤖 **LangGraph Agent**: Features dynamic date & semester prompt injection, conversation memory summarization (>12 messages), and mandatory tool calling (`rag_answer`).
- ⚡ **Full Lifecycle Observability**: Console logging traces every step (`[UPLOAD]` → `[TABLE-AWARE LOAD]` → `[STORE]` → `[BM25 SYNC]` → `[RETRIEVAL]` → `[RERANK]` → `[AGENT]` → `[RESPONSE]`).

### 🎨 Institutional Frontend Redesign
- 🏛️ **Academic Slate-Blue Aesthetic**: Restrained palette featuring Slate 950 (`#0b0f19`), Slate 900 (`#111827`), and Royal Blue (`#2563eb`).
- ✒️ **Editorial Typography**: Inter for clean UI components; Lora serif for readable reading text.
- 💬 **Interactive Features**: Pulsing terminal block cursor streaming, mobile drawer menu, document status modal, and academic prompt starters.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Redesigned Frontend UI                     │
│         (Slate-Blue / Inter & Lora Typography)          │
└───────────────────────────┬─────────────────────────────┘
                            │ (SSE Stream / JSON)
                            ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI Server (main.py)                  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Agent (agent.py)                 │
│  - System Prompt (Date & Semester aware)                │
│  - Summary Checkpointer (>12 messages)                  │
└───────────────────────────┬─────────────────────────────┘
                            │ (Tool Call)
                            ▼
┌─────────────────────────────────────────────────────────┐
│           Hybrid Search System (hybrid_search.py)       │
│  ┌───────────────────────┐   ┌───────────────────────┐  │
│  │ ChromaDB (Dense Vector)│   │  BM25 Index (Sparse)  │  │
│  └───────────┬───────────┘   └───────────┬───────────┘  │
│              └────────────┬──────────────┘              │
│                           ▼                             │
│       Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               Context-Grounded LLM Response             │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Agent Framework** | LangGraph | Stateful multi-turn orchestration with checkpointer |
| **LLM Provider** | Groq (`llama-3.3-70b-versatile`) | Fast inference LLM via Groq API |
| **PDF Extraction** | `pymupdf4llm` | PDF rendering into GitHub-Flavored Markdown |
| **Dense Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | 384-dimensional dense text vectors |
| **Sparse Index** | `rank_bm25` (BM25Okapi) | Frequency-weighted term matching |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Pairwise relevance reranking |
| **Vector Database** | ChromaDB | Persistent HNSW vector store (`vector_store_v2`) |
| **API Framework** | FastAPI & Uvicorn | Async HTTP server with streaming support |
| **Frontend** | Vanilla JS / CSS3 | Modern Slate-Blue design with Inter & Lora typography |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Anaconda / Miniconda (recommended)
- Groq API Key ([console.groq.com](https://console.groq.com))

### Step 1: Clone Repository
```bash
git clone https://github.com/balajichedurupalli0630-oss/Agentic_ai.git
cd Agentic_ai
```

### Step 2: Create Conda Environment
```bash
conda create -n langchain312 python=3.12 -y
conda activate langchain312
```

### Step 3: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 4: Environment Configuration
Create a `.env` file in the `backend/` directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
VECTOR_STORE_PATH=./vector_store_v2
```

---

## ⚙️ Configuration

### 1. Vector Store Settings (`vector_base.py`)
```python
vectorstore = VectorStore(
    collection_name="pdf_document",
    persist_directory="./vector_store_v2"
)
```

### 2. Embedding Model (`embeddings.py`)
```python
embedding_loader = Embeddings(
    model_name="all-MiniLM-L6-v2"
)
```

### 3. Cross-Encoder Reranker (`cross_encoder.py`)
```python
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

### 4. Table Chunking Logic (`doc_load_chunk_tables.py`)
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=160,
    separators=["\n\n", "\n\n\n", "\n", ". ", " ", ""],
)
```

---

## 🚀 Usage

### 1. Start the Backend Server
```bash
cd backend
python main.py
```
*Server will start at:* `http://0.0.0.0:8000`

### 2. Launch the Web Interface
Open `frontend/index.html` directly in your browser or run a simple local web server:
```bash
python -m http.server 3000 --directory frontend/
```
*UI will be live at:* `http://localhost:3000`

---

## 📚 API Documentation

### Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/health` | System readiness & document counts |
| **POST** | `/upload` | Table-aware document ingestion (`.pdf`, `.txt`, `.csv`) |
| **POST** | `/chat` | Non-streaming agent conversation |
| **POST** | `/chat/stream` | Server-Sent Events (SSE) streaming chat |
| **GET** | `/documents` | List all indexed documents with source metadata |
| **GET** | `/documents/stats` | Detailed index sync statistics |
| **DELETE**| `/documents/clear` | Clear all documents from vector store and BM25 index |

---

### Endpoint Specifications

#### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "online",
  "documents": 199,
  "bm25_synced": true,
  "cross_encoder": true
}
```

---

#### 2. Upload Document (Table-Aware)
```http
POST /upload
```
**Request (Multipart Form):**
- `file`: PDF / TXT / CSV file
- `use_contextual`: `false` (default)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/Academic_Calendar.pdf"
```

**Response:**
```json
{
  "file_name": "Academic_Calendar.pdf",
  "chunks_created": 46,
  "total_documents": 199,
  "contextualized": false,
  "message": "Document processed successfully"
}
```

---

#### 3. Chat (Non-Streaming)
```http
POST /chat
```
**Request Body:**
```json
{
  "message": "What is the start date for academic registration?",
  "session_id": "user_session_123"
}
```

**Response:**
```json
{
  "response": "According to the Academic Calendar, academic registration commences on 02.07.2026.",
  "session_id": "user_session_123"
}
```

---

#### 4. Chat Streaming (SSE)
```http
POST /chat/stream
```
**Request Body:**
```json
{
  "message": "What are the key midterm exam dates?",
  "session_id": "user_session_123"
}
```
**Response Stream:**
```
data: {"type": "content", "content": "Midterm "}
data: {"type": "content", "content": "examinations "}
data: {"type": "content", "content": "are "}
...
```

---

#### 5. List Documents
```http
GET /documents
```
**Response:**
```json
{
  "total_chunks": 199,
  "total_sources": 2,
  "documents": [
    {
      "source": "Attention is All you Need.pdf",
      "chunks": 153
    },
    {
      "source": "Academic Calendar AY2026-27.pdf",
      "chunks": 46
    }
  ]
}
```

---

## 📂 Project Structure

```
Agentic_ai/
│
├── backend/
│   ├── main.py                    # FastAPI application & REST routing
│   ├── agent.py                   # LangGraph Agent, prompts & memory checkpointer
│   ├── doc_load_chunk.py          # Central loader orchestrator
│   ├── doc_load_chunk_tables.py   # pymupdf4llm loader & Markdown table parsing
│   ├── hybrid_search.py           # Hybrid retrieval (ChromaDB + BM25 + CrossEncoder)
│   ├── bm25.py                    # BM25Okapi sparse search index
│   ├── cross_encoder.py           # SentenceTransformers CrossEncoder wrapper
│   ├── vector_base.py             # ChromaDB client & MD5 dedup store
│   ├── embeddings.py              # SentenceTransformers embedding loader
│   ├── tools.py                   # rag_answer tool implementation
│   ├── llm.py                     # Groq LLM initialization
│   ├── state.py                   # AgentState Pydantic schema
│   ├── auth.py                    # API key verification dependency
│   ├── requirements.txt           # Backend python dependencies
│   └── vector_store_v2/           # ChromaDB database store
│
├── frontend/
│   └── index.html                 # Redesigned academic assistant UI
│
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

---

## 🔍 How It Works

### 1. Table-Aware PDF Ingestion
```
PDF File → pymupdf4llm.to_markdown() → Split Blank Lines
                                            │
           ┌────────────────────────────────┴────────────────┐
           ▼                                                 ▼
   is_table_block == True                         is_table_block == False
   (Atomic Table Chunk)                           (Recursive Text Chunking)
   content_type="table"                           content_type="text"
           │                                                 │
           └────────────────────────────────┬────────────────┘
                                            ▼
                           Store in ChromaDB & BM25 Index
```

### 2. Retrieval & Reranking Pipeline
```
User Query ──► Hybrid Retrieval (Alpha=0.5) ──► Top 15 Candidates ──► Cross-Encoder ──► Top 5 Chunks ──► LLM Context
               - Dense Vector Search (0.50)                          Reranking
               - BM25 Keyword Search (0.50)                          (ms-marco)
```

---

## 🚀 Advanced Features

### 1. Time-Aware Query System Prompt
The system automatically detects the current month and injects the corresponding academic period into the agent prompt (e.g. *Even Semester 2026* vs *Odd Semester 2026*).

### 2. Conversation Summarization
When conversation history exceeds `12` messages, `summarize_node` compresses older messages into a summary while preserving the most recent turn.

---

## 🐛 Troubleshooting

### Issue 1: `DuplicateIDError` on Upload
- **Cause**: Identical chunk text occurring within a single document batch.
- **Solution**: Handled automatically in `vector_base.py` via `raw_key = f"{src}_{i}_{content}"`.

### Issue 2: `[Errno 48] Address already in use`
- **Cause**: Existing process running on port `8000`.
- **Solution**: Run `kill -9 $(lsof -t -i:8000)` and restart `python main.py`.

---

## 🧪 Testing & Observability

Run the end-to-end verification script:
```bash
python scratch/test_e2e.py
```

Console logs trace the pipeline lifecycle:
```text
[UPLOAD START] Received upload request for file: 'Academic Calendar AY2026-27.pdf'
[TABLE-AWARE LOAD] Academic Calendar AY2026-27.pdf: 46 chunks (20 table, 26 text)
[STORE SUCCESS] Added 46 unique chunks to ChromaDB collection 'pdf_document'
[BM25 SYNC SUCCESS] Synced 199 documents to BM25 index.
[AGENT START] Invoking LLM with tool bindings...
[RETRIEVAL CANDIDATES] Found 15 raw candidates
[RETRIEVAL RERANK SUCCESS] Top candidate score: 5.0041
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open a Pull Request or submit an Issue.

---

**Built with ❤️ for high-precision document Q&A and table reasoning.**
