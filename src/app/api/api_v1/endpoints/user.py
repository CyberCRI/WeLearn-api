from datetime import datetime, timedelta
import uuid

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.sql import select

from src.app.models.db_models import InferredUser, Session
from src.app.services.sql_db import session_maker
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)


@router.post(
    "/user",
    summary="creates new user",
    description="Create a new user in the user db",
    response_model=dict,
)
def handle_user(user_id: uuid.UUID | None = None):
    """
    Create a new user in the user db
    If user_id is provided, check if the user exists in the db
    If the user exists, return the user id
    If the user does not exist, create a new user and return the user id
    """
    try:
        if user_id:
            statement = select(InferredUser.id).where(InferredUser.id == user_id)
            with session_maker() as s:
                user = s.execute(statement).first()
            if user:
                logger.info(f"User {user_id} already exists")
                return {"user_id": user.id}
        logger.info(f"User {user_id} does not exist, creating new user")
        with session_maker() as s:
            user = InferredUser()
            s.add(user)
            s.commit()
            logger.info(f"User {user.id} created")
            return {"user_id": user.id}
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating user: {e}")


@router.post(
    "/session",
    summary="creates new session",
    description="Create a new session in the user db",
    response_model=dict,
)
def handle_session(user_id: uuid.UUID, request: Request, session_id: uuid.UUID | None = None):
    """
    Create a new session in the session db
    If session_id is provided, check if the session exists in the db and if the end_at is still more recent than time now
    If the session exists and the end_at is still more recent than time now, return the session id
    If the session exists and the end_at is not more recent than time now, create a new session and return the session id
    If the session does not exist, create a new session and return the session id
    """
    try:
        # Check if user exists
        with session_maker() as s:
            user = s.execute(
                select(InferredUser.id).where(InferredUser.id == user_id)
            ).first()
        if not user:
            logger.error(f"User={user_id} does not exist")
            raise HTTPException(
                status_code=404, detail=f"User={user_id} does not exist"
            )

        # If session_id is provided, check if a valid session exists
        if session_id:
            with session_maker() as s:
                session = s.execute(
                    select(Session.id).where(
                        (Session.id == session_id)
                        & (Session.inferred_user_id == user_id)
                        & (Session.end_at > datetime.now())
                    )
                ).first()
            if session:
                logger.info(f"Session={session_id} user={user_id} already exists")
                return {"session_id": session.id}

        # Create a new session if not found or expired
        logger.info(
            f"Session={session_id} user={user_id} does not exist or is expired, creating new session"
        )
        now = datetime.now()
        with session_maker() as s:
            new_session = Session(
                inferred_user_id=user_id,
                created_at=now,
                end_at=now + timedelta(hours=24),
                host=getattr(request.client, "host", None) if request.client else None,
            )
            s.add(new_session)
            s.commit()
            logger.info(f"Session={new_session.id} user={user_id} created")
            return {"session_id": new_session.id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {e}")
