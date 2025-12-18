# src/app/api/api_v1/api.py

from fastapi import APIRouter

from src.app.api.api_v1.endpoints import (
    chat,
    metric,
    micro_learning,
    search,
    tutor,
    user,
)

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/qna", tags=["qna"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(tutor.router, prefix="/tutor", tags=["tutor"])
api_router.include_router(metric.router, prefix="/metric", tags=["metric"])
api_router.include_router(
    micro_learning.router, prefix="/micro_learning", tags=["micro_learning"]
)
api_router.include_router(user.router, prefix="/user", tags=["user"])


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
]
