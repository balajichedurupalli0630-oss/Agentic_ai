"""
router.py

API v1 Router aggregator mounting health, chat, and document endpoints.
"""

from fastapi import APIRouter
from api.v1.health import router as health_router
from api.v1.chat import router as chat_router
from api.v1.documents import router as documents_router, upload_document

v1_router = APIRouter(prefix="/v1")

# Mount endpoints
v1_router.include_router(health_router)
v1_router.include_router(chat_router)
v1_router.include_router(documents_router)

# Also expose /v1/upload for convenience
v1_router.add_api_route(
    "/upload",
    upload_document,
    methods=["POST"],
    tags=["Documents"],
    summary="Upload document (alias)",
)
