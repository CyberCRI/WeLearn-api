# user api endpoints
# /user
# /session
from fastapi import APIRouter, HTTPException
from sqlalchemy.sql import select
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)

@router.get("/user", summary="creates new user", description="Create a new user in the user db", response_model=dict)
async def handle_user():
    try:
        # check if id and if id in table
        # put new line in user db
        # return user id
        return {"user_id": "user_id"}
    except Exception as e:
        logger.error(f"Error creating user: {e}")


@router.get("/session", summary="creates new session", description="Create a new session in the user db", response_model=dict)
async def handle_session():
    try:
        # check if id and if id in table
        # check if end_at is still more recent than time now
        # put new line in session db
        # return session id
        return {"session_id": "session_id"}
    except Exception as e:
        logger.error(f"Error creating session: {e}")
