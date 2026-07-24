"""
auth.py

Simple API key authentication via X-API-Key header.
If API_KEY env var is not set, all requests pass through (open mode).
Set API_KEY in your .env to enable protection.
"""

import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """
    FastAPI dependency — raises 403 if API_KEY is set and request doesn't match.
    Pass as: Depends(verify_api_key)
    """
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        # No key configured → open access (dev mode)
        return

    if not api_key or api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )