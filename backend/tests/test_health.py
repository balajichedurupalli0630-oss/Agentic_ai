"""
test_health.py

Automated integration tests for root system status and v1 health endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root GET status response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online"
    assert "v1_api" in data
    assert "X-Request-ID" in response.headers


def test_v1_health_endpoint():
    """Verify API v1 health check status."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online"
    assert "documents" in data
    assert "X-Request-ID" in response.headers
