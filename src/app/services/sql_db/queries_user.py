import uuid
from datetime import datetime, timedelta

from sqlalchemy.sql import select
from welearn_database.data.models import Bookmark, InferredUser, Session

from src.app.services.sql_service import session_maker
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


# ============================
# Synchronous DB helper functions
# ============================


def get_or_create_user_sync(user_id: uuid.UUID | None = None) -> uuid.UUID:
    with session_maker() as s:
        if user_id:
            user = s.execute(
                select(InferredUser.id).where(InferredUser.id == user_id)
            ).first()
            if user:
                return user.id
        user = InferredUser()
        s.add(user)
        s.commit()
        return user.id


def get_or_create_session_sync(
    user_id: uuid.UUID, session_id: uuid.UUID | None = None, host: str = "unknown"
) -> uuid.UUID:
    now = datetime.now()
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.id == user_id)
        ).first()
        if not user:
            raise ValueError(f"User {user_id} does not exist")

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

        new_session = Session(
            inferred_user_id=user_id,
            created_at=now,
            end_at=now + timedelta(hours=24),
            host=host,
        )
        s.add(new_session)
        s.commit()
        return new_session.id


def get_user_bookmarks_sync(user_id: uuid.UUID) -> list[Bookmark]:
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.id == user_id)
        ).first()
        if not user:
            raise ValueError(f"User {user_id} does not exist")
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
            raise ValueError(f"User {user_id} does not exist")
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
            raise ValueError(f"User {user_id} does not exist")
        bookmark = s.execute(
            select(Bookmark).where(
                (Bookmark.inferred_user_id == user_id)
                & (Bookmark.document_id == document_id)
            )
        ).first()
        if not bookmark:
            raise ValueError(
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
            raise ValueError(f"User {user_id} does not exist")
        bookmark = s.execute(
            select(Bookmark).where(
                (Bookmark.inferred_user_id == user_id)
                & (Bookmark.document_id == document_id)
            )
        ).first()
        if bookmark:
            raise ValueError(
                f"Bookmark {document_id} for user {user_id} already exists"
            )
        new_bookmark = Bookmark(document_id=document_id, inferred_user_id=user_id)
        s.add(new_bookmark)
        s.commit()
        return document_id
