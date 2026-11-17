# src/app/core/lifespan.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from qdrant_client import AsyncQdrantClient

from src.app.core.langsmith import configure_langsmith_tracing
from src.app.search.services.search import preload_embedding_models
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
    # the API answers no request (/health/ included) until this returns, so the
    # pod only becomes ready once the models are in memory
    await run_in_threadpool(
        preload_embedding_models, settings.EMBEDDING_MODELS_TO_PRELOAD
    )

    yield
    await app.state.qdrant.close()
    await close_chat_model()
