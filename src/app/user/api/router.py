import uuid

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from src.app.shared.api.errors import retry_later
from src.app.shared.domain.constants import SESSION_COOKIE_NAME, SESSION_TTL_SECONDS
from src.app.shared.domain.responses import ErrorResponse
from src.app.shared.infra.rate_limit import SlidingWindowRateLimiter, client_ip
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.shared.utils.requests import (
    extract_origin_from_request,
    extract_session_cookie,
)
from src.app.user.domain.models import (
    AccountAccessResponse,
    BookmarkAddedResponse,
    BookmarkDeletedResponse,
    BookmarksDeletedResponse,
    BookmarksResponse,
    GuestFeatureQuota,
    GuestQuotaResponse,
    InstitutionData,
    InstitutionDataResponse,
    LogoutResponse,
    SignUpRequest,
    SignUpResponse,
    UniversityRequest,
    UniversityRequestResponse,
    UserAndSessionResponse,
)
from src.app.user.infra.keycloak import (
    KeycloakConfigError,
    KeycloakUnavailableError,
    extract_bearer_token,
)
from src.app.user.infra.queries import (
    add_institution_data_to_user_sync,
    add_user_bookmark_sync,
    delete_user_bookmark_sync,
    delete_user_bookmarks_sync,
    end_session_sync,
    get_user_bookmarks_sync,
)
from src.app.user.services.account import (
    ACCOUNT_REQUIRED_RESPONSES,
    BEARER_TOKEN_RESPONSES,
    account_required,
    guest_restrictions_enabled,
    keycloak_errors_as_http,
    verify_account_token,
)
from src.app.user.services.guest_quota import (
    GUEST_QUOTA_RESPONSES,
    NEW_GUEST_RATE_LIMITED_RESPONSES,
    GuestFeature,
    enforce_new_guest_ip_limit,
    quota_status,
    resolve_quota_user,
)
from src.app.user.services.limited_access import get_account_access
from src.app.user.services.session import (
    resolve_keycloak_user_and_session,
    resolve_user_and_session,
)
from src.app.user.services.signup import sign_up_user
from src.app.user.services.university_request import send_university_request

router = APIRouter()
logger = logger_utils(__name__)
settings = get_settings()

# Sign-up limits, per pod/process (see shared/infra/rate_limit.py). The IP is
# the user's only where FORWARDED_ALLOW_IPS lists the ingress pods; elsewhere it
# is the ingress' address shared by every user and the per-IP limit is a loose
# flood ceiling. The per-email one stops repeated sign-ups (and Keycloak
# emails) for one address.
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

# requests to add a university: each one emails the WeLearn team
UNIVERSITY_REQUEST_RATE_LIMIT_PER_IP = 5
UNIVERSITY_REQUEST_ACCEPTED_MESSAGE = "Your request has been sent to the WeLearn team."
university_request_ip_limiter = SlidingWindowRateLimiter(
    UNIVERSITY_REQUEST_RATE_LIMIT_PER_IP, SIGNUP_RATE_WINDOW_SECONDS
)


def enforce_signup_rate_limit(request: Request, email: str) -> None:
    """429 RATE_LIMITED, with Retry-After, past the per-IP or per-email limit."""
    retry_after = signup_ip_limiter.hit(client_ip(request)) or (
        signup_email_limiter.hit(email)
    )
    if retry_after:
        logger.warning("Sign-up rate limited. retry_after=%.0fs", retry_after)
        raise retry_later(
            status.HTTP_429_TOO_MANY_REQUESTS,
            {
                "message": "Too many sign-up attempts, try again later",
                "code": "RATE_LIMITED",
            },
            retry_after,
        )


def enforce_university_request_rate_limit(request: Request) -> None:
    """429 RATE_LIMITED, with Retry-After, past the per-IP limit."""
    retry_after = university_request_ip_limiter.hit(client_ip(request))
    if retry_after:
        logger.warning(
            "University request rate limited. retry_after=%.0fs", retry_after
        )
        raise retry_later(
            status.HTTP_429_TOO_MANY_REQUESTS,
            {
                "message": "Too many requests, try again later",
                "code": "RATE_LIMITED",
            },
            retry_after,
        )


async def resolve_bookmark_user(request: Request) -> uuid.UUID:
    """
    Resolve the bookmarks owner: from the Keycloak token when there is one, so
    bookmarks follow the account across sessions, else from the session cookie.
    With guest restrictions on, bookmarks need an account: no cookie fallback.
    """
    session_uuid = extract_session_cookie(request)
    token = extract_bearer_token(request)
    if not token and guest_restrictions_enabled():
        raise account_required()
    if not token and not session_uuid:
        raise HTTPException(status_code=401, detail="Session cookie is missing")
    host = extract_origin_from_request(request)

    with keycloak_errors_as_http():
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
    return user_id


@router.post(
    "/user_and_session",
    summary="Create new user and session",
    response_model=UserAndSessionResponse,
    responses={**BEARER_TOKEN_RESPONSES, **NEW_GUEST_RATE_LIMITED_RESPONSES},
)
async def handle_user_and_session(
    request: Request, response: Response, referer: str | None = None
):
    host = extract_origin_from_request(request)
    session_uuid = extract_session_cookie(request)
    token = extract_bearer_token(request)

    with keycloak_errors_as_http():
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
                before_new_guest=lambda: enforce_new_guest_ip_limit(request),
            )

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
    "/logout",
    summary="End the current session",
    response_model=LogoutResponse,
)
async def logout(request: Request, response: Response):
    """
    End the caller's session and drop its cookie, so that after a Keycloak
    logout the browser is no longer identified as the account user: the next
    /user_and_session without a token gets a new guest. Answers 200 with or
    without a session.
    """
    session_uuid = extract_session_cookie(request)
    if session_uuid:
        await run_in_threadpool(end_session_sync, session_uuid)

    # same attributes as set_cookie in /user_and_session, otherwise the
    # browser keeps the cookie (delete_cookie defaults to samesite="lax")
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        domain=settings.SESSION_COOKIE_DOMAIN,
        secure=True,
        httponly=True,
        samesite=None,
    )

    return {"message": "logged out"}


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
            locale=data.locale,
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


@router.post(
    "/university-request",
    summary="Ask to add a university to the login page",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UniversityRequestResponse,
    responses={
        429: {
            "model": ErrorResponse,
            "description": "Too many requests (code RATE_LIMITED), see Retry-After",
        },
        500: {
            "model": ErrorResponse,
            "description": "Sending emails is not configured (code EMAIL_DISABLED)",
        },
        503: {
            "model": ErrorResponse,
            "description": "The email could not be sent (code EMAIL_UNAVAILABLE)",
        },
    },
)
async def request_university(request: Request, data: UniversityRequest):
    """
    Email the WeLearn team a university missing from the login page's list of
    identity providers. No session or account needed.
    """
    enforce_university_request_rate_limit(request)
    await send_university_request(data)
    return {"message": UNIVERSITY_REQUEST_ACCEPTED_MESSAGE}


@router.get(
    "/bookmarks",
    summary="Get user bookmarks",
    response_model=BookmarksResponse,
    responses=ACCOUNT_REQUIRED_RESPONSES,
)
async def get_user_bookmarks(request: Request):
    user_id = await resolve_bookmark_user(request)

    bookmarks = await run_in_threadpool(get_user_bookmarks_sync, user_id)
    return {"bookmarks": bookmarks}


@router.delete(
    "/bookmarks",
    summary="Delete all user bookmarks",
    response_model=BookmarksDeletedResponse,
    responses=ACCOUNT_REQUIRED_RESPONSES,
)
async def delete_user_bookmarks(request: Request):
    user_id = await resolve_bookmark_user(request)

    deleted_count = await run_in_threadpool(delete_user_bookmarks_sync, user_id)
    return {"deleted": deleted_count}


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
    responses=ACCOUNT_REQUIRED_RESPONSES,
)
@router.delete(
    LEGACY_BOOKMARK_PATH,
    summary="Delete a user bookmark (deprecated, use /bookmarks/{document_id})",
    response_model=BookmarkDeletedResponse,
    responses=ACCOUNT_REQUIRED_RESPONSES,
    deprecated=True,
    name="delete_user_bookmark_legacy",
)
async def delete_user_bookmark(request: Request, document_id: uuid.UUID):
    user_id = await resolve_bookmark_user(request)

    deleted_id = await run_in_threadpool(
        delete_user_bookmark_sync, user_id, document_id
    )
    return {"deleted": deleted_id}


@router.post(
    "/bookmarks/{document_id}",
    summary="Add user bookmark",
    response_model=BookmarkAddedResponse,
    responses=ACCOUNT_REQUIRED_RESPONSES,
)
@router.post(
    LEGACY_BOOKMARK_PATH,
    summary="Add user bookmark (deprecated, use /bookmarks/{document_id})",
    response_model=BookmarkAddedResponse,
    responses=ACCOUNT_REQUIRED_RESPONSES,
    deprecated=True,
    name="add_user_bookmark_legacy",
)
async def add_user_bookmark(request: Request, document_id: uuid.UUID):
    user_id = await resolve_bookmark_user(request)

    added_id = await run_in_threadpool(add_user_bookmark_sync, user_id, document_id)
    return {"added": added_id}


@router.get(
    "/access",
    summary="How long the account's access lasts",
    response_model=AccountAccessResponse,
    responses={
        **BEARER_TOKEN_RESPONSES,
        401: {
            "model": ErrorResponse,
            "description": (
                "No bearer token (code ACCOUNT_REQUIRED), or invalid or expired token"
            ),
        },
    },
)
async def get_access(request: Request) -> AccountAccessResponse:
    """
    Whether the caller's account is time-limited (Google or email/password
    login) and until when. Needs a bearer token; an account whose access has
    ended gets 403 ACCESS_EXPIRED.
    """
    account = await verify_account_token(request)
    if account is None:
        raise account_required()
    return await get_account_access(account)


@router.get(
    "/guest_quota",
    summary="Daily guest quotas left",
    response_model=GuestQuotaResponse,
    responses={
        **BEARER_TOKEN_RESPONSES,
        401: GUEST_QUOTA_RESPONSES[401],
    },
)
async def get_guest_quota(request: Request) -> GuestQuotaResponse:
    """
    What the caller has left today of each guest quota, without using any.
    Not limited (an account, or guest restrictions off): `limited` is false.
    """
    user_id = await resolve_quota_user(request)
    if user_id is None:
        return GuestQuotaResponse(limited=False)

    async def feature_quota(feature: GuestFeature) -> GuestFeatureQuota:
        limit, remaining, resets_in = await quota_status(feature, user_id)
        return GuestFeatureQuota(limit=limit, remaining=remaining, resets_in=resets_in)

    return GuestQuotaResponse(
        limited=True,
        chat=await feature_quota("chat"),
        syllabus=await feature_quota("syllabus"),
        feedback=await feature_quota("feedback"),
        files=await feature_quota("files"),
        adaptation=await feature_quota("adaptation"),
    )


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

    user_id, _ = await resolve_user_and_session(
        session_uuid=session_uuid,
        host=host,
        referer=None,
    )
    await run_in_threadpool(
        add_institution_data_to_user_sync, user_id, data.institution, data.role
    )

    return {
        "message": "Institution data added to user",
        "institution": data.institution,
        "role": data.role,
    }
