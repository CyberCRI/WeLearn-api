# src/app/router.py

from fastapi import APIRouter

from src.app.bibliography.api import router as bibliography_router
from src.app.chat.api import router as chat_router
from src.app.metric.api import router as metric_router
from src.app.micro_learning.api import router as micro_learning_router
from src.app.search.api import router as search_router
from src.app.tutor.api import router as tutor_router
from src.app.user.api import router as user_router

api_router = APIRouter()
api_router.include_router(chat_router.router, prefix="/qna", tags=["qna"])
api_router.include_router(search_router.router, prefix="/search", tags=["search"])
api_router.include_router(tutor_router.router, prefix="/tutor", tags=["tutor"])
api_router.include_router(metric_router.router, prefix="/metric", tags=["metric"])
api_router.include_router(
    micro_learning_router.router, prefix="/micro_learning", tags=["micro_learning"]
)
api_router.include_router(user_router.router, prefix="/user", tags=["user"])
api_router.include_router(
    bibliography_router.router, prefix="/bibliography", tags=["bibliography"]
)


api_tags_metadata = [
    {
        "name": "search",
        "description": "Search operations",
    },
    {
        "name": "qna",
        "description": "Q&A operations with openai",
    },
    {
        "name": "tutor",
        "description": "Tests for tutor operations",
    },
    {
        "name": "micro_learning",
        "description": "Micro learning journey operations",
    },
    {
        "name": "user",
        "description": "User operations",
    },
    {
        "name": "metric",
        "description": "Metric information",
    },
    {
        "name": "bibliography",
        "description": "Bibliography exporters for documents collected by welearn",
    },
]
