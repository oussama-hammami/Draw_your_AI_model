"""Tests for the /health endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_returns_200() -> None:
    """GET /health must return HTTP 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_body() -> None:
    """GET /health must return correct JSON structure."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data
    assert isinstance(data["timestamp"], float)


@pytest.mark.asyncio
async def test_health_content_type() -> None:
    """GET /health must return application/json content type."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.headers["content-type"] == "application/json"
