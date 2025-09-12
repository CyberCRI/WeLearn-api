# user api endpoints
# /user
# /session
from fastapi import APIRouter, HTTPException
from sqlalchemy.sql import select
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)

@router.post("/user", summary="creates new user", description="Create a new user in the user db", response_model=dict)
async def create_user():
    try:
        # put new line in user db
        # return user id
