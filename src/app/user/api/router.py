import math
import uuid

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from src.app.shared.domain.constants import SESSION_COOKIE_NAME, SESSION_TTL_SECONDS
from src.app.shared.domain.exceptions import UserNotFoundError, WeLearnError
from src.app.shared.domain.responses import ErrorResponse
from src.app.shared.infra.rate_limit import SlidingWindowRateLimiter, client_ip
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.shared.utils.requests import (
    extract_origin_from_request,
    extract_session_cookie,
)
from src.app.user.domain.models import (
    BookmarkAddedResponse,
    BookmarkDeletedResponse,
    BookmarksDeletedResponse,
    BookmarksResponse,
    InstitutionData,
    InstitutionDataResponse,
    SignUpRequest,
    SignUpResponse,
    UserAndSessionResponse,
)
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakTokenError,
    KeycloakUnavailableError,
    extract_bearer_token,
)
from src.app.user.infra.queries import (
    add_institution_data_to_user_sync,
    add_user_bookmark_sync,
    delete_user_bookmark_sync,
    delete_user_bookmarks_sync,
    get_user_bookmarks_sync,
)
from src.app.user.services.session import (
    resolve_keycloak_user_and_session,
    resolve_user_and_session,
)
from src.app.user.services.signup import sign_up_user

router = APIRouter()
logger = logger_utils(__name__)
settings = get_settings()

# Sign-up limits, per pod/process (see shared/infra/rate_limit.py). The IP may
# be the ingress' address shared by every user (uvicorn runs without
# --proxy-headers), so the per-IP limit is a loose flood ceiling; the per-email
# one stops repeated sign-ups (and Keycloak emails) for one address.
SIGNUP_RATE_WINDOW_SECONDS = 3600
SIGNUP_RATE_LIMIT_PER_IP = 20
SIGNUP_RATE_LIMIT_PER_EMAIL = 3
SIGNUP_ACCEPTED_MESSAGE = (
    "If no account exists yet for this email, you will receive an email "
    "to confirm it and set your password."
)

signup_ip_limiter = SlidingWindowRateLimiter(
    SIGNUP_RATE_LIMIT_PER_IP, SIGNUP_RATE_WINDOW_SECONDS
)
signup_email_limiter = SlidingWindowRateLimiter(
    SIGNUP_RATE_LIMIT_PER_EMAIL, SIGNUP_RATE_WINDOW_SECONDS
)


def enforce_signup_rate_limit(request: Request, email: str) -> None:
    """429 RATE_LIMITED, with Retry-After, past the per-IP or per-email limit."""
    retry_after = signup_ip_limiter.hit(client_ip(request)) or (
        signup_email_limiter.hit(email)
    )
    if retry_after:
        logger.warning("Sign-up rate limited. retry_after=%.0fs", retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": "Too many sign-up attempts, try again later",
                "code": "RATE_LIMITED",
            },
            headers={"Retry-After": str(math.ceil(retry_after))},
        )


async def resolve_bookmark_user(request: Request) -> uuid.UUID:
    """
    Resolve the bookmarks owner: from the Keycloak token when there is one, so
    bookmarks follow the account across sessions, else from the session cookie.
    """
    session_uuid = extract_session_cookie(request)
    token = extract_bearer_token(request)
    if not token and not session_uuid:
        raise HTTPException(status_code=401, detail="Session cookie is missing")
    host = extract_origin_from_request(request)

    try:
        if token:
            user_id, _ = await resolve_keycloak_user_and_session(
                token=token,
                session_uuid=session_uuid,
                host=host,
                referer=None,
            )
        else:
            user_id, _ = await resolve_user_and_session(
                session_uuid=session_uuid,
                host=host,
                referer=None,
            )
    except KeycloakTokenError as e:
        logger.warning(f"Invalid keycloak token: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    except KeycloakUnavailableError as e:
        logger.error(f"Keycloak unavailable: {e}")
        raise HTTPException(status_code=503, detail="Keycloak unavailable")
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return user_id


@router.post(
    "/user_and_session",
    summary="Create new user and session",
    response_model=UserAndSessionResponse,
)
async def handle_user_and_session(
    request: Request, response: Response, referer: str | None = None
):
    host = extract_origin_from_request(request)
    session_uuid = extract_session_cookie(request)
    token = extract_bearer_token(request)

    try:
        if token:
            user_id, session_uuid = await resolve_keycloak_user_and_session(
                token=token,
                session_uuid=session_uuid,
                host=host,
                referer=referer,
            )
        else:
            user_id, session_uuid = await resolve_user_and_session(
                session_uuid=session_uuid,
                host=host,
                referer=referer,
                renew_expired_guest=True,
            )
    except KeycloakTokenError as e:
        logger.warning(f"Invalid keycloak token: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    except KeycloakUnavailableError as e:
        logger.error(f"Keycloak unavailable: {e}")
        raise HTTPException(status_code=503, detail="Keycloak unavailable")
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except WeLearnError:
        raise
    except Exception as e:
        logger.error(f"Error creating user and session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=str(session_uuid),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite=None,
        secure=True,
        domain=settings.SESSION_COOKIE_DOMAIN,
    )

    return {
        "message": "session created",
        "session_id": session_uuid,
        "user_id": user_id,
    }


@router.post(
    "/signup",
    summary="Sign up with Keycloak",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SignUpResponse,
    responses={
        429: {
            "model": ErrorResponse,
            "description": "Too many attempts (code RATE_LIMITED), see Retry-After",
        },
        500: {
            "model": ErrorResponse,
            "description": "Sign-up is not configured (code SIGNUP_DISABLED)",
        },
        503: {
            "model": ErrorResponse,
            "description": "Keycloak unavailable (code KEYCLOAK_UNAVAILABLE)",
        },
    },
)
async def sign_up(request: Request, data: SignUpRequest, referer: str | None = None):
    """
    Create a Keycloak account and its WeLearn user. Keycloak emails a link to
    verify the email and set a password. The answer is the same 202 whether the
    email is new or already has an account, so it does not reveal which emails
    are registered.
    """
    enforce_signup_rate_limit(request, data.email)
    session_uuid = extract_session_cookie(request)

    try:
        await sign_up_user(
            email=data.email,
            first_name=data.first_name,
            last_name=data.last_name,
            session_uuid=session_uuid,
            referer=referer,
        )
    except KeycloakUnavailableError as e:
        logger.error("Keycloak unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Keycloak unavailable", "code": "KEYCLOAK_UNAVAILABLE"},
        )
    except KeycloakConfigError as e:
        logger.error("Keycloak misconfigured: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Sign-up is not configured", "code": "SIGNUP_DISABLED"},
        )

    return {"message": SIGNUP_ACCEPTED_MESSAGE}


@router.get(
    "/bookmarks", summary="Get user bookmarks", response_model=BookmarksResponse
)
async def get_user_bookmarks(request: Request):
    user_id = await resolve_bookmark_user(request)

    try:
        bookmarks = await run_in_threadpool(get_user_bookmarks_sync, user_id)
        return {"bookmarks": bookmarks}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except WeLearnError:
        raise
    except Exception as e:
        logger.error(f"Error fetching bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/bookmarks",
    summary="Delete all user bookmarks",
    response_model=BookmarksDeletedResponse,
)
async def delete_user_bookmarks(request: Request):
    user_id = await resolve_bookmark_user(request)

    try:
        deleted_count = await run_in_threadpool(delete_user_bookmarks_sync, user_id)
        return {"deleted": deleted_count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except WeLearnError:
        raise
    except Exception as e:
        logger.error(f"Error deleting bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Each bookmark operation has two routes sharing one handler:
# - `/bookmarks/{document_id}`, the document id in the path;
# - the legacy `/bookmarks/:document_id?document_id=...`, Express-style syntax
#   that FastAPI registers as a literal `:document_id` segment, the id being
#   read from the query. Deprecated, with its own `name` (hence operation id).
#   Remove it once the deployed WeLearn-client calls `/bookmarks/{document_id}`.
# The legacy decorator must stay the inner one: it is then registered first,
# otherwise `{document_id}` would match the literal `:document_id` and 422.
LEGACY_BOOKMARK_PATH = "/bookmarks/:document_id"


@router.delete(
    "/bookmarks/{document_id}",
    summary="Delete a user bookmark",
    response_model=BookmarkDeletedResponse,
)
@router.delete(
    LEGACY_BOOKMARK_PATH,
    summary="Delete a user bookmark (deprecated, use /bookmarks/{document_id})",
    response_model=BookmarkDeletedResponse,
    deprecated=True,
    name="delete_user_bookmark_legacy",
)
async def delete_user_bookmark(request: Request, document_id: uuid.UUID):
    user_id = await resolve_bookmark_user(request)

    try:
        deleted_id = await run_in_threadpool(
            delete_user_bookmark_sync, user_id, document_id
        )
        return {"deleted": deleted_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except WeLearnError:
        raise
    except Exception as e:
        logger.error(f"Error deleting bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/bookmarks/{document_id}",
    summary="Add user bookmark",
    response_model=BookmarkAddedResponse,
)
@router.post(
    LEGACY_BOOKMARK_PATH,
    summary="Add user bookmark (deprecated, use /bookmarks/{document_id})",
    response_model=BookmarkAddedResponse,
    deprecated=True,
    name="add_user_bookmark_legacy",
)
async def add_user_bookmark(request: Request, document_id: uuid.UUID):
    user_id = await resolve_bookmark_user(request)

    try:
        added_id = await run_in_threadpool(add_user_bookmark_sync, user_id, document_id)
        return {"added": added_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except WeLearnError:
        raise
    except Exception as e:
        logger.error(f"Error adding bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/institution-data",
    summary="Add institution data to user",
    response_model=InstitutionDataResponse,
)
async def add_institution_data_to_user(request: Request, data: InstitutionData):
    session_uuid = extract_session_cookie(request)
    if not session_uuid:
        raise HTTPException(status_code=401, detail="Session cookie is missing")
    host = extract_origin_from_request(request)

    try:
        user_id, _ = await resolve_user_and_session(
            session_uuid=session_uuid,
            host=host,
            referer=None,
        )
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        resp = await run_in_threadpool(
            add_institution_data_to_user_sync, user_id, data.institution, data.role
        )

        print(f"add_institution_data_to_user_sync response: {resp}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except WeLearnError:
        raise
    except Exception as e:
        logger.error(f"Error adding institution data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Institution data added to user",
        "institution": data.institution,
        "role": data.role,
    }
