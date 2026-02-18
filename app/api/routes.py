"""API route definitions.

Endpoints:
    GET  /health     — Health check.
    POST /upload     — Upload a diagram file.
    POST /parse      — Parse diagram text into IR.
    POST /generate   — Generate model code from IR.
    GET  /download   — Download the last generated code.
"""

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.generator import generate_keras, generate_pytorch
from app.ir.models import IRGraph
from app.llm import OllamaClient
from app.parser import parse_diagram
from app.parser.exceptions import ParserError
from app.validator import validate
from app.validator.exceptions import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session store (maps session_id → artifacts)
_sessions: dict[str, dict[str, Any]] = {}


# ─── Request / Response models ────────────────────────────────────

class ParseRequest(BaseModel):
    """Request body for /parse."""
    diagram: str
    format: str | None = None  # 'mermaid' or 'drawio'; auto-detected if None


class GenerateRequest(BaseModel):
    """Request body for /generate."""
    session_id: str
    framework: str = "pytorch"  # 'pytorch' or 'keras'
    polish: bool = False  # use Ollama to polish output


class ParseResponse(BaseModel):
    """Response body for /parse."""
    session_id: str
    ir: dict[str, Any]
    node_count: int
    edge_count: int


class GenerateResponse(BaseModel):
    """Response body for /generate."""
    session_id: str
    framework: str
    code: str
    polished: bool


# ─── Endpoints ────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
async def health_check() -> dict[str, Any]:
    """Return application health status."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": time.time(),
    }


@router.post("/upload", tags=["pipeline"])
async def upload_diagram(
    file: UploadFile = File(...),
    format: str | None = Form(None),
) -> ParseResponse:
    """Upload a diagram file, parse it, validate the IR, and return it.

    Accepts .mmd (Mermaid) or .drawio / .xml (Draw.io) files.
    """
    content = (await file.read()).decode("utf-8")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return _parse_and_store(content, format)


@router.post("/parse", tags=["pipeline"])
async def parse_diagram_text(req: ParseRequest) -> ParseResponse:
    """Parse diagram text (inline) into IR JSON."""
    return _parse_and_store(req.diagram, req.format)


@router.post("/generate", tags=["pipeline"])
async def generate_code(req: GenerateRequest) -> GenerateResponse:
    """Generate model code from a previously parsed IR.

    Requires a valid session_id from a prior /upload or /parse call.
    """
    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found")

    graph: IRGraph = session["graph"]

    if req.framework == "pytorch":
        code = generate_pytorch(graph)
    elif req.framework == "keras":
        code = generate_keras(graph)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown framework '{req.framework}'. Use 'pytorch' or 'keras'.",
        )

    polished = False
    if req.polish:
        ollama = OllamaClient()
        code = await ollama.polish_code(code, framework=req.framework)
        polished = True

    session["code"] = code
    session["framework"] = req.framework

    return GenerateResponse(
        session_id=req.session_id,
        framework=req.framework,
        code=code,
        polished=polished,
    )


@router.get("/download", tags=["pipeline"])
async def download_code(session_id: str) -> dict[str, Any]:
    """Download the last generated code for a session.

    Returns the code as a JSON object with filename and content.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    code = session.get("code")
    if code is None:
        raise HTTPException(
            status_code=400,
            detail="No code generated yet. Call /generate first.",
        )

    framework = session.get("framework", "pytorch")
    filename = f"generated_model_{'py' if framework == 'pytorch' else 'keras'}.py"

    return {
        "session_id": session_id,
        "filename": filename,
        "content": code,
    }


# ─── Internal helpers ─────────────────────────────────────────────

def _parse_and_store(content: str, fmt: str | None) -> ParseResponse:
    """Parse diagram content, validate, store in session, return response."""
    try:
        graph = parse_diagram(content, fmt=fmt)
    except ParserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        validate(graph)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = {"graph": graph, "raw": content}

    return ParseResponse(
        session_id=session_id,
        ir=graph.to_dict(),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
    )
