"""
health.py

API v1 health status and retrieval diagnostics endpoint.
"""

from fastapi import APIRouter, HTTPException
from dependencies import get_retrieval_system
from logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Health check endpoint checking retrieval subsystem state."""
    try:
        retrieval_sys = get_retrieval_system()
        stats = retrieval_sys.get_stats()
        return {
            "status": "online",
            "documents": stats.get("total_documents", 0),
            "bm25_synced": stats.get("indices_synced", False),
            "cross_encoder": stats.get("cross_encoder_available", False),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")
