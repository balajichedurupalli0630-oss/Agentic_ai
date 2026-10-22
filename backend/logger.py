"""
logger.py

Structured JSON logging module with request correlation ID context.
Provides ContextVar tracing across async request lifecycles.
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config import settings

# Context variable to hold correlation ID per async execution context
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Retrieve the active correlation ID or return empty string."""
    return correlation_id_var.get() or ""


def set_correlation_id(cid: str) -> None:
    """Set the active correlation ID."""
    correlation_id_var.set(cid)


class JSONFormatter(logging.Formatter):
    """Custom logging formatter outputting structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        cid = get_correlation_id()
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": cid or getattr(record, "correlation_id", None),
            "module": record.module,
            "func_name": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if provided
        for key, val in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName", "correlation_id"
            }:
                log_obj[key] = val

        return json.dumps(log_obj)


def setup_logging():
    """Configure system-wide structured JSON logging."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ"))

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Silence overly verbose external loggers
    for logger_name in ("uvicorn", "uvicorn.access", "chromadb", "httpx", "httpcore"):
        ext_logger = logging.getLogger(logger_name)
        ext_logger.handlers = [handler]
        ext_logger.propagate = False


def get_logger(name: str = __name__) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware to manage X-Request-ID header and correlation context."""

    async def dispatch(self, request: Request, call_next) -> Response:
        header_cid = request.headers.get("X-Request-ID")
        correlation_id = header_cid or f"req_{uuid.uuid4().hex[:12]}"
        set_correlation_id(correlation_id)

        start_time = time.perf_counter()
        logger = get_logger("backend.http")

        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={"http_method": request.method, "path": request.url.path}
        )

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            response.headers["X-Request-ID"] = correlation_id
            logger.info(
                f"Completed request: {request.method} {request.url.path} {response.status_code} in {duration_ms}ms",
                extra={
                    "http_method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            )
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"Unhandled exception in request {request.method} {request.url.path}: {exc}",
                extra={
                    "http_method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
                exc_info=True
            )
            raise exc
