"""
main.py

FastAPI application entrypoint featuring:
- Centralized Pydantic Settings
- Structured JSON Logging & Request Correlation ID Middleware
- Unified JSON Exception Handlers
- Modular API Versioning (/v1)
- Hybrid Retrieval + Cross-Encoder Reranking
- Rate Limiting & Security Controls
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from config import settings
from logger import setup_logging, get_logger, CorrelationIdMiddleware
from exceptions import (
    http_exception_handler,
    validation_exception_handler,
    rate_limit_exception_handler,
    unhandled_exception_handler,
)
from dependencies import set_app_dependencies, get_retrieval_system
from hybrid_search import HybridRetrieval
from embeddings import embedding_loader
from vector_base import vectorstore
from agent import build_agent
from tools import set_retrieval_system
from api.v1.router import v1_router

# Initialize system-wide structured JSON logging
setup_logging()
logger = get_logger("backend.main")

# Rate limiter instance
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.APP_NAME,
    description="Hybrid Retrieval · Cross-Encoder Reranking · SSE Streaming",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter

# Middleware stack
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(404, http_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


# Initialize retrieval subsystem & agent graph
try:
    logger.info("Initializing HybridRetrieval system...")
    retrieval_system = HybridRetrieval(
        vectorstore=vectorstore,
        embedding_loader=embedding_loader,
    )
    logger.info("✓ HybridRetrieval ready")

    set_retrieval_system(retrieval_system)
    logger.info("✓ Tools wired")

    agent_graph = build_agent()
    logger.info("✓ Agent graph built")

    set_app_dependencies(retrieval_system, agent_graph)
    logger.info("✓ Application dependencies wired")
except Exception as e:
    logger.error(f"Initialization failure: {e}", exc_info=True)
    raise

# Mount API v1 router
app.include_router(v1_router)

# Mount top-level aliases for frontend compatibility (/chat, /upload, /documents)
from api.v1.chat import router as chat_router
from api.v1.documents import router as documents_router
app.include_router(chat_router)
app.include_router(documents_router)

# Mount static frontend application if available
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists() and (frontend_dir / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")



# Root & Legacy Health Aliases
@app.get("/", tags=["System"])
async def root():
    return {
        "status": "online",
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "v1_api": "/v1",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
async def health_alias():
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
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )