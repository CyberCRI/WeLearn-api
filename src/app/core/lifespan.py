# src/app/core/lifespan.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from src.app.shared.infra.llm_proxy import LLMProxy
from src.app.shared.utils.dependencies import get_settings
from src.app.tutor.service.tutor import close_chat_model, init_chat_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_chat_model(settings)
    app.state.qdrant = AsyncQdrantClient(
        url=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=100,
    )
    app.state.llm = LLMProxy(
        model=settings.LLM_MODEL_NAME,
        api_key=settings.AZURE_APIM_API_KEY,
        api_base=settings.AZURE_APIM_API_BASE,
        api_version="2024-05-01-preview",
        is_azure_model=True,
    )

    yield
    await app.state.qdrant.close()
    await app.state.llm.close_client()
    await close_chat_model()
