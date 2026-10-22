"""
test_chat_api.py

Automated integration tests for API v1 chat and document endpoints.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_structured_error_response_404():
    """Verify that 404 routes return a unified JSON error envelope with request_id."""
    response = client.get("/v1/nonexistent_route")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert "request_id" in data


def test_document_list_endpoint():
    """Verify listing indexed documents."""
    response = client.get("/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "documents" in data
