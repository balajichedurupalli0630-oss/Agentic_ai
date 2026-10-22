"""
documents.py

API v1 Document management endpoints (upload, list, stats, clear, delete).
"""

import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from auth import verify_api_key
from dependencies import get_retrieval_system
from doc_load_chunk import load_single_document, split_documents
from logger import get_logger
from vector_base import vectorstore

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentUploadResponse(BaseModel):
    file_name: str
    chunks_created: int
    message: str
    total_documents: int
    contextualized: bool


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    use_contextual: bool = False,
    use_llm_context: bool = False,
    _auth=Depends(verify_api_key),
):
    """Upload and index a document into the hybrid search system."""
    logger.info(f"Processing upload for file '{file.filename}' (contextual={use_contextual})")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".txt", ".csv"}:
        logger.warning(f"Rejected upload for file '{file.filename}': extension {ext} not allowed.")
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        documents = load_single_document(temp_path, original_filename=file.filename)
        chunks = split_documents(documents)

        retrieval_sys = get_retrieval_system()
        stats = await retrieval_sys.add_documents(
            documents=chunks,
            use_contextual=use_contextual,
            use_llm_context=use_llm_context,
        )

        logger.info(f"Successfully indexed '{file.filename}'. Total chunks added: {stats.get('total_chunks', len(chunks))}")
        return DocumentUploadResponse(
            file_name=file.filename,
            chunks_created=stats.get("total_chunks", len(chunks)),
            total_documents=vectorstore.collection.count(),
            contextualized=use_contextual,
            message="Document processed and indexed successfully",
        )
    except Exception as e:
        logger.error(f"Error processing upload for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.get("", summary="List documents")
async def list_documents(_auth=Depends(verify_api_key)):
    """List all indexed documents with metadata."""
    try:
        data = vectorstore.collection.get(include=["metadatas"])
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])

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
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Get storage stats")
async def document_stats(_auth=Depends(verify_api_key)):
    """Get system and retrieval index statistics."""
    try:
        retrieval_sys = get_retrieval_system()
        return retrieval_sys.get_stats()
    except Exception as e:
        logger.error(f"Failed to fetch document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", summary="Clear all documents")
async def clear_documents(_auth=Depends(verify_api_key)):
    """Clear all document chunks from vector store and BM25 index."""
    try:
        retrieval_sys = get_retrieval_system()
        retrieval_sys.clear()
        logger.info("Cleared all documents from vector store and BM25 index.")
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}", summary="Delete document by ID")
async def delete_document(doc_id: str, _auth=Depends(verify_api_key)):
    """Delete a document chunk by vector store ID."""
    try:
        vectorstore.delete_document(doc_id)
        logger.info(f"Deleted document chunk {doc_id}.")
        return {"message": f"Document chunk {doc_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete document chunk {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
