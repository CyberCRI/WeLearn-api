# src/app/user/infra/queries.py

import uuid
from datetime import datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import select
from welearn_database.data.models import Bookmark, InferredUser, Session

from src.app.shared.domain.exceptions import (
    BookmarkAlreadyExistsError,
    BookmarkNotFoundError,
    SessionNotFoundError,
    UserNotFoundError,
)
from src.app.shared.infra.db import session_maker
from src.app.shared.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


# ============================
# Synchronous DB helper functions
# ============================


def get_or_create_user_sync(
    user_id: uuid.UUID | None = None, referer: str | None = None
) -> uuid.UUID:
    with session_maker() as s:
        if user_id:
            user = s.execute(
                select(InferredUser.id).where(InferredUser.id == user_id)
            ).first()
            if user:
                return user.id

        logger.info("Creating new user with referer: %s", referer)
        user = InferredUser(origin_referrer=referer)
        s.add(user)
        s.commit()
        return user.id


def get_or_create_user_from_keycloak_sync(
    keycloak_id: uuid.UUID,
    anonymous_user_id: uuid.UUID | None = None,
    referer: str | None = None,
) -> uuid.UUID:
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.keycloak_id == keycloak_id)
        ).first()
        if user:
            return user.id

        anonymous_user = None
        if anonymous_user_id:
            anonymous_user = s.execute(
                select(InferredUser).where(
                    (InferredUser.id == anonymous_user_id)
                    & (InferredUser.keycloak_id.is_(None))
                )
            ).scalar_one_or_none()

        if anonymous_user:
            # keep the anonymous user's history (bookmarks, chats...) on login
            logger.info(
                "Linking user_id=%s to keycloak_id=%s", anonymous_user.id, keycloak_id
            )
            anonymous_user.keycloak_id = keycloak_id
            user = anonymous_user
        else:
            logger.info("Creating new user for keycloak_id=%s", keycloak_id)
            user = InferredUser(keycloak_id=keycloak_id, origin_referrer=referer)
        s.add(user)
        try:
            s.commit()
        except IntegrityError:
            # keycloak_id is unique: a concurrent request created it first
            s.rollback()
            return s.execute(
                select(InferredUser.id).where(InferredUser.keycloak_id == keycloak_id)
            ).scalar_one()
        return user.id


def get_or_create_session_sync(
    user_id: uuid.UUID,
    session_id: uuid.UUID | None = None,
    host: str = "unknown",
    referer: str | None = None,
) -> uuid.UUID:
    now = datetime.now()
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.id == user_id)
        ).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} does not exist")

        if session_id:
            session = s.execute(
                select(Session.id).where(
                    (Session.id == session_id)
                    & (Session.inferred_user_id == user_id)
                    & (Session.end_at > now)
                )
            ).first()
            if session:
                return session.id

        logger.info(
            "Creating new session for user_id=%s with referer: %s", user_id, referer
        )
        new_session = Session(
            inferred_user_id=user_id,
            origin_referrer=referer,
            created_at=now,
            end_at=now + timedelta(hours=24),
            host=host,
        )
        s.add(new_session)
        s.commit()
        return new_session.id


def end_session_sync(session_id: uuid.UUID) -> None:
    """
    End a session now (logout). An unknown or already ended session is left
    as it is: its end is never pushed back.
    """
    now = datetime.now()
    with session_maker() as s:
        session = s.execute(
            select(Session).where((Session.id == session_id) & (Session.end_at > now))
        ).scalar_one_or_none()
        if session is None:
            logger.info("No active session to end. session_id=%s", session_id)
            return
        session.end_at = now
        s.commit()
        logger.info("Session ended. session_id=%s", session_id)


def get_user_from_session_id(session_id: uuid.UUID | None) -> uuid.UUID | None:
    """
    The user of a session that is still valid. An unknown or expired session
    raises SessionNotFoundError: an old cookie must not act as a credential.
    """
    if not session_id:
        return None

    with session_maker() as s:
        user_id = s.execute(
            select(Session.inferred_user_id).where(
                (Session.id == session_id) & (Session.end_at > datetime.now())
            )
        ).scalar_one_or_none()
    if user_id is None:
        raise SessionNotFoundError(f"Session with id {session_id} not found")
    logger.info("Valid session. user_id=%s session_id=%s", user_id, session_id)
    return user_id


def get_user_for_session_renewal(session_id: uuid.UUID | None) -> uuid.UUID | None:
    """
    Like get_user_from_session_id, but an expired session of an anonymous user
    (no keycloak_id) still resolves, so it can be rotated into a new session:
    the cookie outlives the 24h DB session and a guest must keep their user.
    An expired session of a Keycloak-linked user does not: getting back to an
    account takes a token. Only for the /user/user_and_session cookie path.
    """
    if not session_id:
        return None

    with session_maker() as s:
        row = s.execute(
            select(Session.inferred_user_id, Session.end_at, InferredUser.keycloak_id)
            .join(InferredUser, InferredUser.id == Session.inferred_user_id)
            .where(Session.id == session_id)
        ).first()
    if row is None:
        raise SessionNotFoundError(f"Session with id {session_id} not found")
    if row.end_at > datetime.now():
        return row.inferred_user_id
    if row.keycloak_id is None:
        logger.info(
            "Renewing expired guest session. user_id=%s session_id=%s",
            row.inferred_user_id,
            session_id,
        )
        return row.inferred_user_id
    logger.info("Expired session of a keycloak user. session_id=%s", session_id)
    raise SessionNotFoundError(f"Session with id {session_id} has expired")


def get_user_bookmarks_sync(user_id: uuid.UUID) -> list[Bookmark]:
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.id == user_id)
        ).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} does not exist")
        bookmarks = s.execute(
            select(Bookmark).where(Bookmark.inferred_user_id == user_id)
        ).all()
        return [b[0] for b in bookmarks]


def delete_user_bookmarks_sync(user_id: uuid.UUID) -> int:
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.id == user_id)
        ).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} does not exist")
        deleted = (
            s.query(Bookmark).filter(Bookmark.inferred_user_id == user_id).delete()
        )
        s.commit()
        return deleted


def delete_user_bookmark_sync(user_id: uuid.UUID, document_id: uuid.UUID) -> uuid.UUID:
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.id == user_id)
        ).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} does not exist")
        bookmark = s.execute(
            select(Bookmark).where(
                (Bookmark.inferred_user_id == user_id)
                & (Bookmark.document_id == document_id)
            )
        ).first()
        if not bookmark:
            raise BookmarkNotFoundError(
                f"Bookmark {document_id} for user {user_id} does not exist"
            )
        s.delete(bookmark[0])
        s.commit()
        return document_id


def add_user_bookmark_sync(user_id: uuid.UUID, document_id: uuid.UUID) -> uuid.UUID:
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.id == user_id)
        ).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} does not exist")
        bookmark = s.execute(
            select(Bookmark).where(
                (Bookmark.inferred_user_id == user_id)
                & (Bookmark.document_id == document_id)
            )
        ).first()
        if bookmark:
            raise BookmarkAlreadyExistsError(
                f"Bookmark {document_id} for user {user_id} already exists"
            )
        new_bookmark = Bookmark(document_id=document_id, inferred_user_id=user_id)
        s.add(new_bookmark)
        s.commit()
        return document_id


def add_institution_data_to_user_sync(
    user_id: uuid.UUID, institution: str, role: str
) -> uuid.UUID:
    with session_maker() as s:
        user = s.execute(select(InferredUser).where(InferredUser.id == user_id)).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} does not exist")
        user = user[0]
        user.university_title = institution
        user.role = role
        s.add(user)
        s.commit()
        return user_id
