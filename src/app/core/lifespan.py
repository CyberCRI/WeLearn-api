# src/app/core/lifespan.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from src.app.core.langsmith import configure_langsmith_tracing
from src.app.shared.utils.dependencies import get_settings
from src.app.tutor.services.tutor import close_chat_model, init_chat_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_langsmith_tracing(settings)
    await init_chat_model(settings)
    app.state.qdrant = AsyncQdrantClient(
        url=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=100,
    )

    yield
    await app.state.qdrant.close()
    await close_chat_model()
