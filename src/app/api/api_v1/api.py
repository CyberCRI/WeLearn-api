from fastapi import APIRouter

from src.app.api.api_v1.endpoints import chat, search, tutor

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/qna", tags=["qna"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(tutor.router, prefix="/tutor", tags=["tutor"])


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
]
