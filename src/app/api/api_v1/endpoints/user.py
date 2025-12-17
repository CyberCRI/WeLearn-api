import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from src.app.services.sql_db.queries_user import (
    add_user_bookmark_sync,
    delete_user_bookmark_sync,
    delete_user_bookmarks_sync,
    get_or_create_session_sync,
    get_or_create_user_sync,
    get_user_bookmarks_sync,
)
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)


@router.post("/user", summary="Create new user", response_model=dict)
async def handle_user(user_id: uuid.UUID | None = None):
    try:
        user_id = await run_in_threadpool(get_or_create_user_sync, user_id)
        return {"user_id": user_id}
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", summary="Create new session", response_model=dict)
async def handle_session(
    user_id: uuid.UUID, request: Request, session_id: uuid.UUID | None = None
):
    try:
        host = request.headers.get("origin", "unknown")
        session_id = await run_in_threadpool(
            get_or_create_session_sync, user_id, session_id, host
        )
        return {"session_id": session_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/:user_id/bookmarks", summary="Get user bookmarks", response_model=dict)
async def get_user_bookmarks(user_id: uuid.UUID):
    try:
        bookmarks = await run_in_threadpool(get_user_bookmarks_sync, user_id)
        return {"bookmarks": bookmarks}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/:user_id/bookmarks", summary="Delete all user bookmarks", response_model=dict
)
async def delete_user_bookmarks(user_id: uuid.UUID):
    try:
        deleted_count = await run_in_threadpool(delete_user_bookmarks_sync, user_id)
        return {"deleted": deleted_count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/:user_id/bookmarks/:document_id",
    summary="Delete a user bookmark",
    response_model=dict,
)
async def delete_user_bookmark(user_id: uuid.UUID, document_id: uuid.UUID):
    try:
        deleted_id = await run_in_threadpool(
            delete_user_bookmark_sync, user_id, document_id
        )
        return {"deleted": deleted_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/:user_id/bookmarks/:document_id", summary="Add user bookmark", response_model=dict
)
async def add_user_bookmark(user_id: uuid.UUID, document_id: uuid.UUID):
    try:
        added_id = await run_in_threadpool(add_user_bookmark_sync, user_id, document_id)
        return {"added": added_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))
