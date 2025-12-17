from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.app.services.search import init_qdrant, close_qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_qdrant()
    yield
    await close_qdrant()
