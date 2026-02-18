"""Main FastAPI application entry point."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router

logger = logging.getLogger(__name__)

STARTUP_TIME: float = 0.0
FRONTEND_DIR = Path(__file__).parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events."""
    global STARTUP_TIME
    STARTUP_TIME = time.time()
    logger.info("Application starting up")
    yield
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Diagram-to-Code Converter",
        description="Convert deep learning model diagrams into executable PyTorch or Keras code.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Serve frontend static files
    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_frontend() -> FileResponse:
            """Serve the main frontend page."""
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    return app


app = create_app()
