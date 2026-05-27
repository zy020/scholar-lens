from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scholar_lens.api.deps import init_dependencies
from scholar_lens.api.middleware import RequestLoggingMiddleware
from scholar_lens.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    setup_logging()
    init_dependencies()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ScholarLens",
        version="0.1.0",
        description="Educational agent system for reading English academic documents",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from scholar_lens.api.routes import config, documents, chat, notes, memory
    app.include_router(config.router, prefix="/api/config", tags=["config"])
    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(notes.router, prefix="/api/notes", tags=["notes"])
    app.include_router(memory.router, prefix="/api/memory", tags=["memory"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
