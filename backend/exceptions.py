"""
exceptions.py

Unified exception handlers and standardized JSON error response schema.
Ensures consistent API responses across all status codes and exception types.
"""

from typing import Any, Optional
from fastapi import Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded

from logger import get_correlation_id, get_logger

logger = get_logger(__name__)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str


def create_error_response(
    status_code: int,
    code: str,
    message: str,
    details: Optional[Any] = None
) -> JSONResponse:
    """Construct a standardized JSONResponse with ErrorResponse body."""
    correlation_id = get_correlation_id()
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details,
        ),
        request_id=correlation_id,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard FastAPI HTTPExceptions."""
    code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "UNPROCESSABLE_ENTITY",
        500: "INTERNAL_SERVER_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
    message = str(exc.detail) if exc.detail else "An HTTP error occurred"
    logger.warning(f"HTTPException {exc.status_code} [{code}]: {message}")
    return create_error_response(exc.status_code, code, message)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic request validation errors."""
    logger.warning(f"Validation error for request {request.url.path}: {exc.errors()}")
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="VALIDATION_ERROR",
        message="Invalid request parameters or body",
        details=exc.errors(),
    )


async def rate_limit_exception_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Handle SlowAPI rate limit exceeded errors."""
    logger.warning(f"Rate limit exceeded for request {request.url.path}")
    return create_error_response(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        code="RATE_LIMIT_EXCEEDED",
        message="Too many requests. Please slow down and try again later.",
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected internal server errors."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred.",
    )
