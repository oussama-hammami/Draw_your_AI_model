"""Tests for the full API pipeline: upload → parse → generate → download."""

import io

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

MERMAID_DIAGRAM = """\
graph TD
    A[Input: shape=784] --> B[Linear: 256]
    B --> C[ReLU]
    C --> D[Linear: 10]
    D --> E[Softmax]
"""

DRAWIO_DIAGRAM = """\
<mxGraphModel>
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="n1" value="Input: shape=784" vertex="1" parent="1">
      <mxGeometry/>
    </mxCell>
    <mxCell id="n2" value="Linear: 10" vertex="1" parent="1">
      <mxGeometry/>
    </mxCell>
    <mxCell id="e1" edge="1" source="n1" target="n2" parent="1"/>
  </root>
</mxGraphModel>
"""


@pytest.fixture
def transport() -> ASGITransport:
    return ASGITransport(app=app)


# ─── /parse endpoint ─────────────────────────────────────────────

class TestParseEndpoint:

    @pytest.mark.asyncio
    async def test_parse_mermaid(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/parse", json={"diagram": MERMAID_DIAGRAM})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["node_count"] == 5
        assert data["edge_count"] == 4

    @pytest.mark.asyncio
    async def test_parse_drawio(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/parse", json={"diagram": DRAWIO_DIAGRAM})
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] == 2

    @pytest.mark.asyncio
    async def test_parse_invalid_diagram_returns_400(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/parse", json={"diagram": "not a diagram"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_parse_empty_diagram_returns_400(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/parse", json={"diagram": ""})
        assert resp.status_code == 400


# ─── /upload endpoint ────────────────────────────────────────────

class TestUploadEndpoint:

    @pytest.mark.asyncio
    async def test_upload_mermaid_file(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/upload",
                files={"file": ("model.mmd", MERMAID_DIAGRAM.encode(), "text/plain")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] == 5

    @pytest.mark.asyncio
    async def test_upload_empty_file_returns_400(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/upload",
                files={"file": ("empty.mmd", b"", "text/plain")},
            )
        assert resp.status_code == 400


# ─── /generate endpoint ──────────────────────────────────────────

class TestGenerateEndpoint:

    @pytest.mark.asyncio
    async def test_generate_pytorch(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First parse
            parse_resp = await client.post("/parse", json={"diagram": MERMAID_DIAGRAM})
            session_id = parse_resp.json()["session_id"]

            # Then generate
            gen_resp = await client.post("/generate", json={
                "session_id": session_id,
                "framework": "pytorch",
            })

        assert gen_resp.status_code == 200
        data = gen_resp.json()
        assert "import torch" in data["code"]
        assert data["framework"] == "pytorch"

    @pytest.mark.asyncio
    async def test_generate_keras(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            parse_resp = await client.post("/parse", json={"diagram": MERMAID_DIAGRAM})
            session_id = parse_resp.json()["session_id"]

            gen_resp = await client.post("/generate", json={
                "session_id": session_id,
                "framework": "keras",
            })

        assert gen_resp.status_code == 200
        assert "tensorflow" in gen_resp.json()["code"]

    @pytest.mark.asyncio
    async def test_generate_invalid_session_returns_404(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/generate", json={
                "session_id": "nonexistent",
                "framework": "pytorch",
            })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_generate_unknown_framework_returns_400(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            parse_resp = await client.post("/parse", json={"diagram": MERMAID_DIAGRAM})
            session_id = parse_resp.json()["session_id"]

            resp = await client.post("/generate", json={
                "session_id": session_id,
                "framework": "jax",
            })
        assert resp.status_code == 400


# ─── /download endpoint ──────────────────────────────────────────

class TestDownloadEndpoint:

    @pytest.mark.asyncio
    async def test_download_after_generate(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            parse_resp = await client.post("/parse", json={"diagram": MERMAID_DIAGRAM})
            session_id = parse_resp.json()["session_id"]

            await client.post("/generate", json={
                "session_id": session_id,
                "framework": "pytorch",
            })

            dl_resp = await client.get("/download", params={"session_id": session_id})

        assert dl_resp.status_code == 200
        data = dl_resp.json()
        assert "content" in data
        assert "import torch" in data["content"]
        assert "filename" in data

    @pytest.mark.asyncio
    async def test_download_without_generate_returns_400(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            parse_resp = await client.post("/parse", json={"diagram": MERMAID_DIAGRAM})
            session_id = parse_resp.json()["session_id"]

            resp = await client.get("/download", params={"session_id": session_id})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_download_invalid_session_returns_404(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/download", params={"session_id": "fake"})
        assert resp.status_code == 404


# ─── Full pipeline flow ──────────────────────────────────────────

class TestFullPipelineFlow:

    @pytest.mark.asyncio
    async def test_upload_parse_generate_download(self, transport: ASGITransport) -> None:
        """Complete flow: upload file → generate pytorch → download."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Upload
            upload_resp = await client.post(
                "/upload",
                files={"file": ("model.mmd", MERMAID_DIAGRAM.encode(), "text/plain")},
            )
            assert upload_resp.status_code == 200
            session_id = upload_resp.json()["session_id"]

            # 2. Generate
            gen_resp = await client.post("/generate", json={
                "session_id": session_id,
                "framework": "pytorch",
            })
            assert gen_resp.status_code == 200
            assert "nn.Linear" in gen_resp.json()["code"]

            # 3. Download
            dl_resp = await client.get("/download", params={"session_id": session_id})
            assert dl_resp.status_code == 200
            assert "import torch" in dl_resp.json()["content"]

    @pytest.mark.asyncio
    async def test_parse_generate_download_keras(self, transport: ASGITransport) -> None:
        """Complete flow via /parse for Keras."""
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            parse_resp = await client.post("/parse", json={"diagram": MERMAID_DIAGRAM})
            session_id = parse_resp.json()["session_id"]

            gen_resp = await client.post("/generate", json={
                "session_id": session_id,
                "framework": "keras",
            })
            assert gen_resp.status_code == 200

            dl_resp = await client.get("/download", params={"session_id": session_id})
            assert dl_resp.status_code == 200
            assert "tensorflow" in dl_resp.json()["content"]
