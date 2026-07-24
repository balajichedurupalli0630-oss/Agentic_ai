# 🤖 RAG Agent - Intelligent & Table-Aware Document Question Answering System

A production-ready **Table-Aware Retrieval-Augmented Generation (RAG)** system powered by **LangGraph** agents, **Hybrid Search (Vector + BM25)**, **Cross-Encoder Reranking**, and **`pymupdf4llm` table parsing** with an institutional academic chat interface.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
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
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## ✨ Features

### 📊 Table-Aware Ingestion & Chunking
- 📄 **`pymupdf4llm` PDF Parsing**: Extracts PDFs into GFM-style Markdown blocks (`| col | col |`).
- 🧩 **Atomic Table Blocks**: Heuristically detects tables (`is_table_block`) and preserves every table as a single, indivisible chunk — preventing mid-row or mid-cell splitting.
- 🏷️ **Metadata Tagging & Dedup**: Tags chunks with `content_type: "table"` or `"text"`, preserves original user filenames, and guarantees batch-level unique MD5 chunk keys to avoid duplicate errors.

### 🧠 Intelligent Hybrid Retrieval & Reranking
- 🔍 **Hybrid Retrieval**: Merges dense semantic vector search (ChromaDB) with sparse keyword search (BM25Okapi).
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

| Component | Technology |
|-----------|-----------|
| **Agent Orchestration** | LangGraph |
| **LLM Provider** | Groq (`llama-3.3-70b-versatile`) |
| **PDF & Table Extraction** | `pymupdf4llm` (GFM Markdown rendering) |
| **Dense Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Sparse Search** | `rank_bm25` (BM25Okapi) |
| **Reranker** | CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **Vector Database** | ChromaDB (`hnsw:space`: cosine) |
| **API Server** | FastAPI & Uvicorn |
| **Frontend** | Vanilla JS / CSS3 (Inter & Lora Google Fonts) |

---

## 📦 Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/balajichedurupalli0630-oss/Agentic_ai.git
cd Agentic_ai
```

### Step 2: Set Up Conda / Virtual Environment
```bash
conda create -n langchain312 python=3.12 -y
conda activate langchain312
```

### Step 3: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 4: Environment Variables
Create a `.env` file inside `backend/`:
```env
GROQ_API_KEY=your_groq_api_key_here
VECTOR_STORE_PATH=./vector_store_v2
```

---

## 🚀 Usage

### 1. Start the Backend Server
```bash
cd backend
python main.py
```
The server will start on **`http://0.0.0.0:8000`**.

### 2. Launch the Web Interface
Open `frontend/index.html` directly in your web browser, or serve it via a static web server:
```bash
python -m http.server 3000 --directory frontend/
```
Navigate to **`http://localhost:3000`**.

---

## 📚 API Endpoints

### 1. Health & Status
```http
GET /health
GET /documents/stats
```
Returns system status, total indexed document counts, BM25 sync status, and cross-encoder availability.

### 2. Upload Document (Table-Aware)
```http
POST /upload
```
Form Data: `file` (`.pdf`, `.txt`, `.csv`). Routes PDFs through `pymupdf4llm` and tags table blocks.

### 3. Chat with Agent
```http
POST /chat
```
Sends user query to the LangGraph agent, executing hybrid retrieval and cross-encoder reranking.

---

## 📂 Project Structure

```
Agentic_ai/
│
├── backend/
│   ├── main.py                    # FastAPI server & route handlers
│   ├── agent.py                   # LangGraph agent graph & prompt logic
│   ├── doc_load_chunk.py          # Central loader router
│   ├── doc_load_chunk_tables.py   # pymupdf4llm table extractor & block splitter
│   ├── hybrid_search.py           # Hybrid retrieval (Vector + BM25 + Rerank)
│   ├── bm25.py                    # BM25Okapi search implementation
│   ├── cross_encoder.py           # Cross-encoder reranking wrapper
│   ├── vector_base.py             # ChromaDB client & MD5 dedup store
│   ├── embeddings.py              # SentenceTransformers embedding loader
│   ├── tools.py                   # rag_answer tool definition
│   ├── llm.py                     # Groq LLM initialization
│   ├── state.py                   # AgentState schema
│   ├── auth.py                    # API key validation helper
│   ├── requirements.txt           # Dependency requirements
│   └── vector_store_v2/           # ChromaDB database directory
│
├── frontend/
│   └── index.html                 # Redesigned academic assistant UI
│
├── .gitignore                     # Excludes cache, vector stores, secrets
└── README.md                      # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

---

**Built for high-precision document Q&A and table reasoning.**
