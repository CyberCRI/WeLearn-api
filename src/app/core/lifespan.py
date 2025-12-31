# src/app/core/lifespan.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from src.app.api.dependencies import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.qdrant = AsyncQdrantClient(
        url=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=100,
    )
    yield
    await app.state.qdrant.close()
