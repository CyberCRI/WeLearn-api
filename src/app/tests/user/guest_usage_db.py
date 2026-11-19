"""
A sqlite database for the guest quotas: the features of the 1_5_7 migration,
the given guests, and the guest usage queries patched onto it.
"""

import unittest
import uuid
from datetime import datetime
from unittest.mock import patch

from welearn_database.data.models import Feature, InferredUser

from src.app.tests.data_collection.sqlite_db import make_sqlite_session_maker
from src.app.user.infra.guest_usage import add_guest_usage_sync, get_guest_usage_sync
from src.app.user.services.guest_quota import (
    GUEST_QUOTA_WINDOW,
    GUEST_USAGE_KEYS,
    GuestFeature,
    guest_quota_limit,
)

FEATURE_TITLES = ["chat", "microlearning", "syllabus"]


def use_guest_usage_db(test: unittest.TestCase, *user_ids: uuid.UUID):
    """Patch a fresh database in for the duration of `test`; returns its sessionmaker."""
    engine, s_maker = make_sqlite_session_maker()
    patcher = patch("src.app.user.infra.guest_usage.session_maker", s_maker)
    patcher.start()
    test.addCleanup(patcher.stop)
    test.addCleanup(engine.dispose)

    with s_maker() as s:
        s.add_all(Feature(title=title) for title in FEATURE_TITLES)
        s.add_all(InferredUser(id=user_id) for user_id in user_ids)
        s.commit()
    return s_maker


def guest_left(feature: GuestFeature, user_id: uuid.UUID) -> int:
    """What the guest has left of `feature` today."""
    used, _ = get_guest_usage_sync(
        user_id, GUEST_USAGE_KEYS[feature], datetime.now() - GUEST_QUOTA_WINDOW
    )
    return guest_quota_limit(feature) - used


def guest_use(feature: GuestFeature, user_id: uuid.UUID, times: int = 1) -> None:
    """Count `times` uses of `feature`, as if the guest had made them now."""
    for _ in range(times):
        now = datetime.now()
        add_guest_usage_sync(
            user_id,
            GUEST_USAGE_KEYS[feature],
            guest_quota_limit(feature),
            now - GUEST_QUOTA_WINDOW,
            now,
        )
