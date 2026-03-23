import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from src.app.services.sql_db.queries_user import (
    add_user_bookmark_sync,
    delete_user_bookmark_sync,
    delete_user_bookmarks_sync,
    get_or_create_session_sync,
    get_or_create_user_sync,
    get_user_bookmarks_sync,
    get_user_from_session_id,
)
from src.app.shared.domain.constants import SESSION_COOKIE_NAME, SESSION_TTL_SECONDS
from src.app.shared.utils.requests import (
    extract_origin_from_request,
    extract_session_cookie,
)
from src.app.user.utils.utils import resolve_user_and_session
from src.app.utils.logger import logger as logger_utils


router = APIRouter()
logger = logger_utils(__name__)


@router.post(
    "/user_and_session", summary="Create new user and session", response_model=dict
)
async def handle_user_and_session(
    request: Request, response: Response, referer: str | None = None
):
    host = extract_origin_from_request(request)
    session_uuid = extract_session_cookie(request)

    _, session_uuid = await resolve_user_and_session(
        session_uuid=session_uuid,
        host=host,
        referer=referer,
    )

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=str(session_uuid),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=False,  #  True in production (HTTPS)
    )

    return {"message": "session created"}


@router.post("/user", summary="Create new user", response_model=dict)
async def handle_user(user_id: uuid.UUID | None = None, referer: str | None = None):
    try:
        user_id = await run_in_threadpool(get_or_create_user_sync, user_id, referer)
        return {"user_id": user_id}
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", summary="Create new session", response_model=dict)
async def handle_session(
    user_id: uuid.UUID,
    request: Request,
    session_id: uuid.UUID | None = None,
    referer: str | None = None,
):
    try:
        host = extract_origin_from_request(request)
        session_id = await run_in_threadpool(
            get_or_create_session_sync, user_id, session_id, host, referer
        )
        return {"session_id": session_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bookmarks", summary="Get user bookmarks")
async def get_user_bookmarks(request: Request):
    session_uuid = extract_session_cookie(request)
    host = extract_origin_from_request(request)

    user_id, _ = await resolve_user_and_session(
        session_uuid=session_uuid,
        host=host,
        referer=None,
    )
    try:
        bookmarks = await run_in_threadpool(get_user_bookmarks_sync, user_id)
        print(f"Bookmarks for user {user_id}: {bookmarks}")
        return {"bookmarks": bookmarks}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/bookmarks", summary="Delete all user bookmarks", response_model=dict)
async def delete_user_bookmarks(request: Request):
    session_uuid = extract_session_cookie(request)
    host = extract_origin_from_request(request)

    user_id, _ = await resolve_user_and_session(
        session_uuid=session_uuid,
        host=host,
        referer=None,
    )
    try:
        deleted_count = await run_in_threadpool(delete_user_bookmarks_sync, user_id)
        return {"deleted": deleted_count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/bookmarks/:document_id",
    summary="Delete a user bookmark",
    response_model=dict,
)
async def delete_user_bookmark(request: Request, document_id: uuid.UUID):
    session_uuid = extract_session_cookie(request)
    host = extract_origin_from_request(request)

    user_id, _ = await resolve_user_and_session(
        session_uuid=session_uuid,
        host=host,
        referer=None,
    )
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
    "/bookmarks/:document_id", summary="Add user bookmark", response_model=dict
)
async def add_user_bookmark(request: Request, document_id: uuid.UUID):
    session_uuid = extract_session_cookie(request)
    host = extract_origin_from_request(request)

    user_id, _ = await resolve_user_and_session(
        session_uuid=session_uuid,
        host=host,
        referer=None,
    )
    try:
        added_id = await run_in_threadpool(add_user_bookmark_sync, user_id, document_id)
        return {"added": added_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))
