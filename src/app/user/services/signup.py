from typing import Optional
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.infra.keycloak import KeycloakUnavailableError
from src.app.user.infra.keycloak_admin import (
    KeycloakUserExistsError,
    create_user,
    find_user_id_by_email,
    send_signup_email,
)
from src.app.user.infra.queries import (
    get_or_create_user_from_keycloak_sync,
    get_user_from_session_id,
)

logger = logger_utils(__name__)


async def sign_up_user(
    email: str,
    first_name: Optional[str],
    last_name: Optional[str],
    session_uuid: Optional[UUID],
    referer: Optional[str],
) -> UUID:
    """
    Create a Keycloak account for the email and the WeLearn user matching it.
    Keycloak emails the user a link to verify the email and set a password.
    Raises KeycloakUserExistsError if the email already has an account.
    """
    if await find_user_id_by_email(email):
        raise KeycloakUserExistsError(f"Keycloak user already exists for {email}")

    keycloak_id = await create_user(email, first_name, last_name)
    logger.info("Created keycloak user. keycloak_id=%s", keycloak_id)

    anonymous_user_id = None
    if session_uuid:
        try:
            anonymous_user_id = await run_in_threadpool(
                get_user_from_session_id, session_uuid
            )
        except SessionNotFoundError:
            logger.info("Session %s not found, ignoring it", session_uuid)

    # The anonymous user is linked only if it has no keycloak_id yet. One that
    # belongs to another account (another email, since this one was free) is
    # left alone and a new user is created instead.
    user_id = await run_in_threadpool(
        get_or_create_user_from_keycloak_sync, keycloak_id, anonymous_user_id, referer
    )

    try:
        await send_signup_email(keycloak_id)
    except KeycloakUnavailableError as e:
        # the account exists: "forgot password" on the login page recovers it
        logger.error("Sign-up email not sent. keycloak_id=%s error=%s", keycloak_id, e)

    return user_id
