import unittest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

from welearn_database.data.models import Bookmark, InferredUser, Session

from src.app.shared.domain.exceptions import (
    BookmarkAlreadyExistsError,
    BookmarkNotFoundError,
    SessionNotFoundError,
    UserNotFoundError,
)
from src.app.tests.data_collection.sqlite_db import make_sqlite_session_maker
from src.app.user.infra import queries


class UserQueriesTestBase(unittest.TestCase):
    def setUp(self):
        self.engine, self.s_maker = make_sqlite_session_maker()
        patcher = patch("src.app.user.infra.queries.session_maker", self.s_maker)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.engine.dispose)

        self.user_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=self.user_id))
            s.commit()

    def _add_session(self, end_at: datetime, user_id: uuid.UUID | None = None):
        session_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(
                Session(
                    id=session_id,
                    inferred_user_id=user_id or self.user_id,
                    end_at=end_at,
                )
            )
            s.commit()
        return session_id

    def _add_keycloak_user(self) -> uuid.UUID:
        user_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=user_id, keycloak_id=uuid.uuid4()))
            s.commit()
        return user_id

    def _add_bookmark(self, document_id: uuid.UUID, user_id: uuid.UUID | None = None):
        with self.s_maker() as s:
            s.add(
                Bookmark(
                    document_id=document_id, inferred_user_id=user_id or self.user_id
                )
            )
            s.commit()

    def _bookmarks(self):
        with self.s_maker() as s:
            return [
                (b.inferred_user_id, b.document_id) for b in s.query(Bookmark).all()
            ]


class TestGetUserFromSessionId(UserQueriesTestBase):
    def test_none_session_id(self):
        self.assertIsNone(queries.get_user_from_session_id(None))

    def test_returns_session_owner(self):
        session_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(
                Session(
                    id=session_id,
                    inferred_user_id=self.user_id,
                    end_at=datetime.now() + timedelta(hours=1),
                )
            )
            s.commit()

        self.assertEqual(queries.get_user_from_session_id(session_id), self.user_id)

    def test_unknown_session_raises(self):
        with self.assertRaises(SessionNotFoundError):
            queries.get_user_from_session_id(uuid.uuid4())


class TestGetOrCreateSession(UserQueriesTestBase):
    def test_unknown_user_raises(self):
        with self.assertRaises(UserNotFoundError):
            queries.get_or_create_session_sync(uuid.uuid4())

    def test_expired_session_is_replaced(self):
        expired = uuid.uuid4()
        with self.s_maker() as s:
            s.add(
                Session(
                    id=expired,
                    inferred_user_id=self.user_id,
                    end_at=datetime.now() - timedelta(hours=1),
                )
            )
            s.commit()

        new_id = queries.get_or_create_session_sync(
            self.user_id, expired, host="h", referer="r"
        )

        self.assertNotEqual(new_id, expired)
        with self.s_maker() as s:
            new_session = s.get(Session, new_id)
            self.assertEqual(new_session.inferred_user_id, self.user_id)
            self.assertEqual(new_session.host, "h")
            self.assertEqual(new_session.origin_referrer, "r")
            self.assertGreater(new_session.end_at, datetime.now())


class TestGetUserBookmarks(UserQueriesTestBase):
    def test_unknown_user_raises(self):
        with self.assertRaises(UserNotFoundError):
            queries.get_user_bookmarks_sync(uuid.uuid4())

    def test_returns_only_user_bookmarks(self):
        other_user = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=other_user))
            s.commit()
        doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
        self._add_bookmark(doc_a)
        self._add_bookmark(doc_b)
        self._add_bookmark(uuid.uuid4(), other_user)

        bookmarks = queries.get_user_bookmarks_sync(self.user_id)

        self.assertEqual({b.document_id for b in bookmarks}, {doc_a, doc_b})
        self.assertTrue(all(isinstance(b, Bookmark) for b in bookmarks))


class TestDeleteUserBookmarks(UserQueriesTestBase):
    def test_unknown_user_raises(self):
        with self.assertRaises(UserNotFoundError):
            queries.delete_user_bookmarks_sync(uuid.uuid4())

    def test_deletes_all_user_bookmarks_and_returns_count(self):
        other_user = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=other_user))
            s.commit()
        other_doc = uuid.uuid4()
        self._add_bookmark(uuid.uuid4())
        self._add_bookmark(uuid.uuid4())
        self._add_bookmark(other_doc, other_user)

        deleted = queries.delete_user_bookmarks_sync(self.user_id)

        self.assertEqual(deleted, 2)
        self.assertEqual(self._bookmarks(), [(other_user, other_doc)])


class TestDeleteUserBookmark(UserQueriesTestBase):
    def test_unknown_user_raises(self):
        with self.assertRaises(UserNotFoundError):
            queries.delete_user_bookmark_sync(uuid.uuid4(), uuid.uuid4())

    def test_missing_bookmark_raises(self):
        with self.assertRaises(BookmarkNotFoundError) as ctx:
            queries.delete_user_bookmark_sync(self.user_id, uuid.uuid4())
        self.assertIn("does not exist", ctx.exception.message)

    def test_deletes_only_the_given_bookmark(self):
        doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
        self._add_bookmark(doc_a)
        self._add_bookmark(doc_b)

        result = queries.delete_user_bookmark_sync(self.user_id, doc_a)

        self.assertEqual(result, doc_a)
        self.assertEqual(self._bookmarks(), [(self.user_id, doc_b)])


class TestAddUserBookmark(UserQueriesTestBase):
    def test_unknown_user_raises(self):
        with self.assertRaises(UserNotFoundError):
            queries.add_user_bookmark_sync(uuid.uuid4(), uuid.uuid4())

    def test_adds_bookmark(self):
        doc_id = uuid.uuid4()

        result = queries.add_user_bookmark_sync(self.user_id, doc_id)

        self.assertEqual(result, doc_id)
        self.assertEqual(self._bookmarks(), [(self.user_id, doc_id)])

    def test_duplicate_bookmark_raises(self):
        doc_id = uuid.uuid4()
        self._add_bookmark(doc_id)

        with self.assertRaises(BookmarkAlreadyExistsError) as ctx:
            queries.add_user_bookmark_sync(self.user_id, doc_id)

        self.assertIn("already exists", ctx.exception.message)
        self.assertEqual(len(self._bookmarks()), 1)


class TestAddInstitutionData(UserQueriesTestBase):
    def test_unknown_user_raises(self):
        with self.assertRaises(UserNotFoundError):
            queries.add_institution_data_to_user_sync(uuid.uuid4(), "LPI", "teacher")

    def test_updates_user(self):
        result = queries.add_institution_data_to_user_sync(
            self.user_id, "LPI", "teacher"
        )

        self.assertEqual(result, self.user_id)
        with self.s_maker() as s:
            user = s.get(InferredUser, self.user_id)
            self.assertEqual(user.university_title, "LPI")
            self.assertEqual(user.role, "teacher")


class TestSessionExpiry(UserQueriesTestBase):
    def test_strict_lookup_rejects_expired_session(self):
        expired = self._add_session(datetime.now() - timedelta(minutes=1))

        with self.assertRaises(SessionNotFoundError):
            queries.get_user_from_session_id(expired)

    def test_strict_lookup_rejects_expired_keycloak_session(self):
        user_id = self._add_keycloak_user()
        expired = self._add_session(datetime.now() - timedelta(days=2), user_id)

        with self.assertRaises(SessionNotFoundError):
            queries.get_user_from_session_id(expired)

    def test_renewal_lookup_none(self):
        self.assertIsNone(queries.get_user_for_session_renewal(None))

    def test_renewal_lookup_unknown_session_raises(self):
        with self.assertRaises(SessionNotFoundError):
            queries.get_user_for_session_renewal(uuid.uuid4())

    def test_renewal_lookup_valid_session(self):
        user_id = self._add_keycloak_user()
        valid = self._add_session(datetime.now() + timedelta(hours=1), user_id)

        self.assertEqual(queries.get_user_for_session_renewal(valid), user_id)

    def test_renewal_lookup_expired_guest_session(self):
        expired = self._add_session(datetime.now() - timedelta(days=30))

        self.assertEqual(queries.get_user_for_session_renewal(expired), self.user_id)

    def test_renewal_lookup_expired_keycloak_session_raises(self):
        user_id = self._add_keycloak_user()
        expired = self._add_session(datetime.now() - timedelta(minutes=1), user_id)

        with self.assertRaises(SessionNotFoundError):
            queries.get_user_for_session_renewal(expired)


class TestEndSession(UserQueriesTestBase):
    def _end_at(self, session_id: uuid.UUID) -> datetime:
        with self.s_maker() as s:
            return s.get(Session, session_id).end_at

    def test_ends_a_valid_session(self):
        valid = self._add_session(datetime.now() + timedelta(hours=1))

        queries.end_session_sync(valid)

        self.assertLessEqual(self._end_at(valid), datetime.now())
        with self.assertRaises(SessionNotFoundError):
            queries.get_user_from_session_id(valid)

    def test_unknown_session_is_a_no_op(self):
        valid = self._add_session(datetime.now() + timedelta(hours=1))
        before = self._end_at(valid)

        queries.end_session_sync(uuid.uuid4())

        self.assertEqual(self._end_at(valid), before)

    def test_does_not_touch_another_session(self):
        ended = self._add_session(datetime.now() + timedelta(hours=1))
        other = self._add_session(datetime.now() + timedelta(hours=1))
        other_end = self._end_at(other)

        queries.end_session_sync(ended)

        self.assertEqual(self._end_at(other), other_end)
        self.assertEqual(queries.get_user_from_session_id(other), self.user_id)

    def test_already_ended_session_is_not_extended(self):
        end_at = datetime.now() - timedelta(days=2)
        expired = self._add_session(end_at)

        queries.end_session_sync(expired)

        self.assertEqual(self._end_at(expired), end_at)

    def test_renewal_lookup_after_logout_of_keycloak_user_raises(self):
        user_id = self._add_keycloak_user()
        valid = self._add_session(datetime.now() + timedelta(hours=1), user_id)

        queries.end_session_sync(valid)

        # getting back to the account takes a token: a new guest is created
        with self.assertRaises(SessionNotFoundError):
            queries.get_user_for_session_renewal(valid)

    def test_renewal_lookup_after_logout_of_guest_keeps_the_user(self):
        valid = self._add_session(datetime.now() + timedelta(hours=1))

        queries.end_session_sync(valid)

        # existing behaviour: an ended guest session is still renewed
        self.assertEqual(queries.get_user_for_session_renewal(valid), self.user_id)
