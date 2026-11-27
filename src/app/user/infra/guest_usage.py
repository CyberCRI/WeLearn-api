# src/app/user/infra/guest_usage.py
"""
The limited actions of guests, one row per use in user_related.guest_feature_usage.
A quota counts the rows of one (feature, action) created since a given time.
"""

import uuid
from datetime import datetime

from sqlalchemy import delete, func, select
from welearn_database.data.models import Feature, GuestFeatureUsage, InferredUser

from src.app.shared.domain.exceptions import UserNotFoundError
from src.app.shared.infra.db import session_maker

# (feature title, action)
UsageKey = tuple[str, str]


def _usage_since(s, user_id: uuid.UUID, key: UsageKey, since: datetime):
    title, action = key
    return s.execute(
        select(func.count(), func.min(GuestFeatureUsage.created_at))
        .join(Feature, Feature.id == GuestFeatureUsage.feature_id)
        .where(
            (GuestFeatureUsage.inferred_user_id == user_id)
            & (Feature.title == title)
            & (GuestFeatureUsage.action == action)
            & (GuestFeatureUsage.created_at > since)
        )
    ).one()


def get_guest_usage_sync(
    user_id: uuid.UUID, key: UsageKey, since: datetime
) -> tuple[int, datetime | None]:
    """(uses of `key` since `since`, the oldest of them)."""
    with session_maker() as s:
        used, oldest = _usage_since(s, user_id, key, since)
        return used, oldest


def _feature_id(s, title: str) -> uuid.UUID:
    feature_id = s.execute(
        select(Feature.id).where(Feature.title == title)
    ).scalar_one_or_none()
    if feature_id is None:
        raise LookupError(f"Feature {title!r} is missing from user_related.feature")
    return feature_id


def _insert_usage(s, user_id: uuid.UUID, key: UsageKey, now: datetime) -> uuid.UUID:
    title, action = key
    usage_id = uuid.uuid4()
    s.add(
        GuestFeatureUsage(
            id=usage_id,
            inferred_user_id=user_id,
            feature_id=_feature_id(s, title),
            action=action,
            created_at=now,
        )
    )
    s.commit()
    return usage_id


def add_guest_usage_sync(
    user_id: uuid.UUID, key: UsageKey, limit: int, since: datetime, now: datetime
) -> tuple[uuid.UUID | None, int, datetime | None]:
    """
    Record one use of `key` unless `limit` are already counted since `since`.
    Returns the id of the new row (None when refused), the uses counted before
    it and the oldest of them. The user's row is locked so that concurrent
    requests of one guest are counted one after the other.
    """
    with session_maker() as s:
        user = s.execute(
            select(InferredUser.id).where(InferredUser.id == user_id).with_for_update()
        ).first()
        if not user:
            raise UserNotFoundError(f"User {user_id} does not exist")

        used, oldest = _usage_since(s, user_id, key, since)
        if used >= limit:
            return None, used, oldest
        return _insert_usage(s, user_id, key, now), used, oldest


def record_guest_event_sync(user_id: uuid.UUID, key: UsageKey, now: datetime) -> None:
    """
    Record one row of `key` without any limit, for an event to count apart
    from the uses (a quota only counts the rows of its own action).
    """
    with session_maker() as s:
        _insert_usage(s, user_id, key, now)


def delete_guest_usage_sync(usage_id: uuid.UUID) -> None:
    with session_maker() as s:
        s.execute(delete(GuestFeatureUsage).where(GuestFeatureUsage.id == usage_id))
        s.commit()
