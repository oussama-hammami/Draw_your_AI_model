"""Smoke tests for the frontend.

Tests that the frontend HTML is served correctly and contains
the expected UI elements. Uses the HTTPX async client (no browser needed).
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def transport() -> ASGITransport:
    return ASGITransport(app=app)


class TestFrontendServing:
    """Tests that the frontend page is served and contains key UI elements."""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_root_returns_html(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert "text/html" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_contains_title(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert "Diagram" in resp.text
        assert "Code" in resp.text

    @pytest.mark.asyncio
    async def test_contains_framework_selector(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert "pytorch" in resp.text.lower()
        assert "keras" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_contains_generate_button(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert "Generate Code" in resp.text

    @pytest.mark.asyncio
    async def test_contains_upload_area(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert "upload" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_contains_download_button(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert "Download" in resp.text

    @pytest.mark.asyncio
    async def test_contains_code_viewer(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert "code-output" in resp.text

    @pytest.mark.asyncio
    async def test_api_still_works_alongside_frontend(self, transport: ASGITransport) -> None:
        """Ensure API endpoints still work after mounting static files."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
