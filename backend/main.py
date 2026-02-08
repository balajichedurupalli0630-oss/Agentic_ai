import os
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import json
import uvicorn
import asyncio

from dotenv import load_dotenv
load_dotenv()
from llm import get_llm
from state import AgentState
from rag import RAG
from embeddings import embedding_loader
from vector_base import vectorstore
from agent import build_agent
from doc_load_chunk import load_documents, load_single_document, split_documents
from langchain_core.messages import HumanMessage

app = FastAPI(
    title="RAG System",
    description="API for Document Question Answering RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = get_llm()
rag_system = RAG(vectorstore, embedding_loader, llm)
agent_graph = build_agent()


class QueryRequest(BaseModel):
    query: str = Field(..., description="Ask questions")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of Documents retrieved")
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity score")
    use_expansion: bool = Field(default=False, description="Enable query expansion")


class ChatRequest(BaseModel):
    message: str = Field(..., description="Users message")
    session_id: str = Field(default="default", description="Session ID for conversation tracking")
    stream: bool = Field(default=False, description="Enable streaming response")


class DocumentUpResponse(BaseModel):
    file_name: str
    chunks_created: int
    message: str
    total_documents: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_count: int


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: Optional[List[Dict[str, Any]]] = None


class StatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    persist_directory: str


class HealthResponse(BaseModel):
    status: str
    message: str


async def process_documents(file_path: str, file_name: str) -> Dict[str, Any]:
    """Process uploaded document: load, chunk, embed, and store"""
    try:
        documents = load_single_document(file_path)
        chunks = split_documents(documents)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_loader.generate_embedding(texts)
        vectorstore.add_documents(chunks, embeddings)

        return {
            "file_name": file_name,
            "chunks_created": len(chunks),
            "total_documents": vectorstore.collections.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health checkup endpoint"""
    return HealthResponse(
        status="online",
        message="RAG System API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_checkup():
    """Detailed health check"""
    try:
        count = vectorstore.collections.count()
        return HealthResponse(
            status="online",
            message=f"System operational. {count} documents in vector store."
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"System Unhealthy: {str(e)}")


@app.post("/upload", response_model=DocumentUpResponse)
async def upload_documents(file: UploadFile = File(...), background_task: BackgroundTasks = None):
    """Upload and process a Document"""
    allowed_extensions = {'.pdf', '.txt', '.csv'}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type: {file_extension} not supported. Allowed types: {allowed_extensions}"
        )
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        result = await process_documents(temp_file_path, file.filename)

        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        return DocumentUpResponse(
            file_name=result['file_name'],
            chunks_created=result['chunks_created'],
            message="Document uploaded and processed successfully",
            total_documents=result['total_documents']
        )
    except Exception as e:
        # Clean up temp file on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with Conversation history using langgraph Agent"""
    try:
        config = {
            "configurable": {
                "thread_id": request.session_id
            }
        }

        input_state = {
    "messages": [HumanMessage(content=request.message)],
    "session_id": request.session_id
}

        result = await agent_graph.ainvoke(input_state, config=config)

        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            response_text = "I couldn't generate any response"
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            sources=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat Failed: {str(e)}")


@app.get("/documents/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector store statistics"""
    try:
        return StatsResponse(
            total_documents=vectorstore.collections.count(),
            collection_name=vectorstore.collection_name,
            persist_directory=vectorstore.persist_directory
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/clear")
async def clear_documents():
    """Clear all documents from vector store"""
    try:
        vectorstore.clear_collection()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document by ID"""
    try:
        vectorstore.delete_document(doc_id)
        return {"message": f"Document {doc_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run Server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )










