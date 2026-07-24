"""
main.py

FastAPI server with:
- Hybrid Retrieval + Cross-Encoder Reranking
- SSE streaming endpoint (/chat/stream)
- Per-IP rate limiting via slowapi
- Optional API key auth via X-API-Key header
"""

import os
import logging
import tempfile
import json

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from auth import verify_api_key
from llm import get_fast_llm
from hybrid_search import HybridRetrieval
from embeddings import embedding_loader
from vector_base import vectorstore
from agent import build_agent
from doc_load_chunk import load_single_document, split_documents
from tools import set_retrieval_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RAG System",
    description="Hybrid Retrieval · Cross-Encoder Reranking · Streaming",
    version="8.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize ────────────────────────────────────────────────────────────

try:
    logger.info("Initializing HybridRetrieval...")
    retrieval_system = HybridRetrieval(
        vectorstore=vectorstore,
        embedding_loader=embedding_loader,
    )
    logger.info("✓ HybridRetrieval ready")

    set_retrieval_system(retrieval_system)
    logger.info("✓ Tools wired")

    agent_graph = build_agent()
    logger.info("✓ Agent ready")
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise

# ── Pydantic models ───────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field(default="default", description="Session ID")


class ChatResponse(BaseModel):
    response: str
    session_id: str


class DocumentUploadResponse(BaseModel):
    file_name: str
    chunks_created: int
    message: str
    total_documents: int
    contextualized: bool


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_input(message: str, session_id: str) -> tuple[dict, dict]:
    """Return (input_state, langgraph_config)"""
    config = {"configurable": {"thread_id": session_id}}
    state = {
        "messages": [{"role": "user", "content": message}],
        "session_id": session_id,
    }
    return state, config


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    return {"status": "online", "version": "8.0.0"}


@app.get("/health")
async def health():
    try:
        stats = retrieval_system.get_stats()
        return {
            "status": "online",
            "documents": stats["total_documents"],
            "bm25_synced": stats["indices_synced"],
            "cross_encoder": stats["cross_encoder_available"],
        }
    except Exception as e:
        raise HTTPException(503, str(e))


# ── Chat (blocking) ───────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    _auth=Depends(verify_api_key),
):
    """Blocking chat — returns full response when done."""
    if vectorstore.collection.count() == 0:
        return ChatResponse(
            response="No documents uploaded. Please upload documents first.",
            session_id=body.session_id,
        )

    try:
        state, config = _build_input(body.message, body.session_id)
        result = await agent_graph.ainvoke(state, config=config)
        messages = result.get("messages", [])
        response_text = (
            messages[-1].content
            if messages and hasattr(messages[-1], "content")
            else "No response generated"
        )
        return ChatResponse(response=response_text, session_id=body.session_id)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, str(e))


# ── Chat (streaming SSE) ──────────────────────────────────────────────────

@app.get("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(
    request: Request,
    message: str,
    session_id: str = "default",
    _auth=Depends(verify_api_key),
):
    """
    Server-Sent Events streaming endpoint.
    
    Each event:  data: {"token": "..."}\n\n
    Final event: data: [DONE]\n\n
    Error event: data: {"error": "..."}\n\n
    
    Usage:
        const es = new EventSource('/chat/stream?message=hello&session_id=abc');
        es.onmessage = e => {
            if (e.data === '[DONE]') { es.close(); return; }
            console.log(JSON.parse(e.data).token);
        };
    """
    if vectorstore.collection.count() == 0:
        async def _empty():
            yield f"data: {json.dumps({'token': 'No documents uploaded. Please upload documents first.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    async def generate():
        state, config = _build_input(message, session_id)
        try:
            async for event in agent_graph.astream_events(state, config=config, version="v2"):
                if event["event"] != "on_chat_model_stream":
                    continue
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node == "tools":
                    continue
                chunk = event["data"].get("chunk")
                if not chunk:
                    continue
                content = chunk.content if hasattr(chunk, "content") else ""
                if not content or not isinstance(content, str):
                    continue
                yield f"data: {json.dumps({'token': content})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering
        },
    )


# ── Documents ─────────────────────────────────────────────────────────────

@app.post("/upload", response_model=DocumentUploadResponse)
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    use_contextual: bool = False,
    use_llm_context: bool = False,
    _auth=Depends(verify_api_key),
):
    print(f"[UPLOAD START] Received upload request for file: '{file.filename}' (contextual={use_contextual})")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".txt", ".csv"}:
        print(f"[UPLOAD REJECTED] File extension {ext} not allowed for '{file.filename}'")
        raise HTTPException(400, f"Unsupported type: {ext}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            content = await file.read()
            f.write(content)
            temp_path = f.name
        
        print(f"[UPLOAD PROGRESS] Saved temp file to {temp_path} ({len(content)} bytes)")
        documents = load_single_document(temp_path, original_filename=file.filename)
        print(f"[UPLOAD PROGRESS] Loaded document source. Splitting into chunks...")
        chunks = split_documents(documents)
        print(f"[UPLOAD PROGRESS] Document split into {len(chunks)} chunks. Indexing into retrieval system...")

        stats = await retrieval_system.add_documents(
            documents=chunks,
            use_contextual=use_contextual,
            use_llm_context=use_llm_context,
        )
        print(f"[UPLOAD SUCCESS] Finished indexing for '{file.filename}'. Added {stats.get('total_chunks', len(chunks))} chunks.")

        return DocumentUploadResponse(
            file_name=file.filename,
            chunks_created=stats["total_chunks"],
            total_documents=vectorstore.collection.count(),
            contextualized=use_contextual,
            message="Document processed successfully",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/documents")
async def list_documents(_auth=Depends(verify_api_key)):
    """List all indexed documents with metadata."""
    try:
        data = vectorstore.collection.get(include=["metadatas"])
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])

        # Group by source file
        sources: dict[str, dict] = {}
        for doc_id, meta in zip(ids, metadatas):
            source = (meta or {}).get("source", "unknown")
            if source not in sources:
                sources[source] = {"source": source, "chunks": 0, "doc_ids": []}
            sources[source]["chunks"] += 1
            sources[source]["doc_ids"].append(doc_id)

        return {
            "total_chunks": len(ids),
            "total_sources": len(sources),
            "documents": list(sources.values()),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/documents/stats")
async def stats(_auth=Depends(verify_api_key)):
    try:
        return retrieval_system.get_stats()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/documents/clear")
async def clear_documents(_auth=Depends(verify_api_key)):
    try:
        retrieval_system.clear()
        return {"message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, _auth=Depends(verify_api_key)):
    try:
        vectorstore.delete_document(doc_id)
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # reload=False in prod/Docker
        log_level="info",
    )