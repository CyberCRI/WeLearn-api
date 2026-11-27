"""
Authentication events: new accounts, logins, logouts and sign-ups, recorded in
user_related.auth_event. The routes run against sqlite with real (test-signed)
Keycloak tokens.
"""

import unittest
import uuid
from datetime import date, datetime, timedelta
from unittest import mock

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from welearn_database.data.enumeration import AuthEventType
from welearn_database.data.models import AuthEvent, InferredUser, Session

from src.app.core.config import settings
from src.app.tests.data_collection.sqlite_db import make_sqlite_session_maker
from src.app.tests.user.test_keycloak import (
    CLIENT_ID,
    KEYCLOAK_REALM,
    KEYCLOAK_URL,
    PRIVATE_KEY,
    make_token,
)
from src.app.user.infra import keycloak, queries
from src.app.user.services import auth_events, limited_access
from src.app.user.services.auth_events import record_auth_event
from src.main import app

BASE = f"{settings.API_V1_STR}/user"
HEADERS = {"X-API-Key": "test"}
SIGNUP = "src.app.user.services.signup"
ACCESS_UNTIL = (date.today() + timedelta(days=30)).isoformat()


class AuthEventsDbTestCase(unittest.TestCase):
    """Queries patched onto a fresh sqlite database."""

    def setUp(self):
        self.engine, self.s_maker = make_sqlite_session_maker()
        patcher = mock.patch.object(queries, "session_maker", self.s_maker)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.engine.dispose)

    def _add_user(self, keycloak_id: uuid.UUID | None = None) -> uuid.UUID:
        user_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=user_id, keycloak_id=keycloak_id))
            s.commit()
        return user_id

    def _add_session(self, user_id: uuid.UUID) -> uuid.UUID:
        session_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(
                Session(
                    id=session_id,
                    inferred_user_id=user_id,
                    end_at=datetime.now() + timedelta(hours=1),
                )
            )
            s.commit()
        return session_id

    def _events(self, event_type: AuthEventType | None = None) -> list[AuthEvent]:
        with self.s_maker() as s:
            query = s.query(AuthEvent)
            if event_type:
                query = query.filter(AuthEvent.event_type == event_type.value)
            events = query.order_by(AuthEvent.created_at).all()
            s.expunge_all()
        return events

    def _event_types(self) -> list[str]:
        return sorted(e.event_type for e in self._events())


class TestAuthEventQueries(AuthEventsDbTestCase):
    def test_new_keycloak_account_is_new_once(self):
        keycloak_id = uuid.uuid4()

        first = queries.get_or_create_user_from_keycloak_sync(keycloak_id)
        second = queries.get_or_create_user_from_keycloak_sync(keycloak_id)

        self.assertTrue(first[1])
        self.assertEqual(second, (first[0], False))

    def test_linked_anonymous_user_is_new_account(self):
        anonymous_id = self._add_user()

        result = queries.get_or_create_user_from_keycloak_sync(
            uuid.uuid4(), anonymous_id
        )

        self.assertEqual(result, (anonymous_id, True))

    def test_one_login_per_keycloak_session(self):
        user_id = self._add_user(uuid.uuid4())

        def add_login(sid):
            return queries.add_auth_event_sync(
                user_id, AuthEventType.LOGIN, None, None, False, sid
            )

        self.assertTrue(add_login("sid-1"))
        self.assertFalse(add_login("sid-1"))
        self.assertTrue(add_login("sid-2"))
        self.assertEqual(
            [e.keycloak_session_id for e in self._events()], ["sid-1", "sid-2"]
        )

    def test_other_events_are_not_deduplicated(self):
        user_id = self._add_user(uuid.uuid4())

        for _ in range(2):
            self.assertTrue(
                queries.add_auth_event_sync(
                    user_id, AuthEventType.LOGOUT, None, "google", True
                )
            )

        self.assertEqual(len(self._events(AuthEventType.LOGOUT)), 2)

    def test_keycloak_user_and_own_session_only(self):
        keycloak_id = uuid.uuid4()
        user_id = self._add_user(keycloak_id)
        own_session = self._add_session(user_id)
        other_session = self._add_session(self._add_user())

        get = queries.get_keycloak_user_and_session_sync
        self.assertEqual(get(keycloak_id, own_session), (user_id, own_session))
        self.assertEqual(get(keycloak_id, other_session), (user_id, None))
        self.assertEqual(get(keycloak_id, None), (user_id, None))
        self.assertEqual(get(uuid.uuid4(), own_session), (None, None))

    def test_no_personal_data_columns(self):
        columns = set(AuthEvent.__table__.columns.keys())
        for personal in ("email", "name", "username", "first_name", "last_name"):
            self.assertNotIn(personal, columns)


class TestRecordAuthEvent(unittest.IsolatedAsyncioTestCase):
    @mock.patch.object(auth_events, "add_auth_event_sync")
    async def test_login_without_sid_is_not_recorded(self, add_mock):
        account = keycloak.KeycloakUser(keycloak_id=uuid.uuid4())

        await record_auth_event(AuthEventType.LOGIN, uuid.uuid4(), account)

        add_mock.assert_not_called()

    @mock.patch.object(auth_events, "add_auth_event_sync")
    async def test_sid_is_only_kept_for_logins(self, add_mock):
        user_id, session_id = uuid.uuid4(), uuid.uuid4()
        account = keycloak.KeycloakUser(
            keycloak_id=uuid.uuid4(),
            identity_provider="univ-paris",
            keycloak_session_id="sid-1",
        )

        await record_auth_event(AuthEventType.LOGOUT, user_id, account, session_id)

        add_mock.assert_called_once_with(
            user_id, AuthEventType.LOGOUT, session_id, "univ-paris", False, None
        )

    @mock.patch.object(auth_events, "add_auth_event_sync")
    async def test_db_failure_is_logged_without_details(self, add_mock):
        add_mock.side_effect = OperationalError("INSERT secret-sql", {}, Exception())
        account = keycloak.KeycloakUser(keycloak_id=uuid.uuid4(), email="a@b.org")

        with self.assertLogs(auth_events.logger.name, level="ERROR") as logs:
            await record_auth_event(AuthEventType.SIGNUP, uuid.uuid4(), account)

        output = "\n".join(logs.output)
        self.assertIn("event_type=signup", output)
        self.assertIn("OperationalError", output)
        self.assertNotIn("secret-sql", output)
        self.assertNotIn("a@b.org", output)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
@mock.patch(
    "src.app.shared.api.monitor_requests.register_endpoint", new=mock.MagicMock()
)
@mock.patch.object(keycloak.settings, "KEYCLOAK_AUDIENCE", None)
@mock.patch.object(keycloak.settings, "KEYCLOAK_CLIENT_ID", CLIENT_ID)
@mock.patch.object(keycloak.settings, "KEYCLOAK_REALM", KEYCLOAK_REALM)
@mock.patch.object(keycloak.settings, "KEYCLOAK_URL", KEYCLOAK_URL)
@mock.patch.object(limited_access.settings, "LIMITED_ACCESS_ENABLED", True)
class TestAuthEventsApi(AuthEventsDbTestCase):
    def setUp(self):
        super().setUp()
        jwks_patcher = mock.patch.object(keycloak, "get_jwks_client")
        jwks_mock = jwks_patcher.start()
        self.addCleanup(jwks_patcher.stop)
        jwks_mock.return_value.get_signing_key_from_jwt.return_value = mock.MagicMock(
            key=PRIVATE_KEY.public_key()
        )
        self.client = TestClient(app)
        self.keycloak_id = uuid.uuid4()

    def _token(self, sid: str | None = "sid-1", provider: str | None = "google"):
        return make_token(
            sub=str(self.keycloak_id),
            sid=sid,
            identity_provider=provider,
            welearn_access_until=ACCESS_UNTIL,
            email="jane@example.org",
            preferred_username="jane",
            given_name="Jane",
            family_name="Doe",
            name="Jane Doe",
        )

    def _user_and_session(self, token: str | None = None, session_id=None):
        headers = dict(HEADERS)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if session_id:
            headers["X-Session-Id"] = str(session_id)
        response = self.client.post(f"{BASE}/user_and_session", headers=headers)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_guest_logging_in_records_new_account_and_login(self):
        guest = self._user_and_session()
        self.assertEqual(self._events(), [])

        account = self._user_and_session(self._token(), guest["session_id"])

        self.assertEqual(account["user_id"], guest["user_id"])
        self.assertEqual(self._event_types(), ["account_created", "login"])
        for event in self._events():
            self.assertEqual(str(event.inferred_user_id), guest["user_id"])
            self.assertEqual(str(event.session_id), guest["session_id"])
            self.assertEqual(event.identity_provider, "google")
            self.assertTrue(event.is_limited_access)
        login = self._events(AuthEventType.LOGIN)[0]
        self.assertEqual(login.keycloak_session_id, "sid-1")
        account_created = self._events(AuthEventType.ACCOUNT_CREATED)[0]
        self.assertIsNone(account_created.keycloak_session_id)

    def test_account_without_guest_session_records_new_account(self):
        account = self._user_and_session(self._token(provider="univ-paris"))

        created = self._events(AuthEventType.ACCOUNT_CREATED)
        self.assertEqual(len(created), 1)
        self.assertEqual(str(created[0].inferred_user_id), account["user_id"])
        self.assertEqual(str(created[0].session_id), account["session_id"])
        self.assertEqual(created[0].identity_provider, "univ-paris")
        self.assertFalse(created[0].is_limited_access)

    def test_known_account_is_not_new(self):
        self._add_user(self.keycloak_id)

        self._user_and_session(self._token())

        self.assertEqual(self._event_types(), ["login"])

    def test_one_login_per_keycloak_session(self):
        first = self._user_and_session(self._token("sid-1"))
        for _ in range(2):
            self._user_and_session(self._token("sid-1"), first["session_id"])
        self._user_and_session(self._token("sid-2"), first["session_id"])

        logins = self._events(AuthEventType.LOGIN)
        self.assertEqual([e.keycloak_session_id for e in logins], ["sid-1", "sid-2"])
        self.assertEqual(len(self._events(AuthEventType.ACCOUNT_CREATED)), 1)

    def test_token_without_sid_records_no_login(self):
        for _ in range(2):
            self._user_and_session(self._token(sid=None))

        self.assertEqual(self._event_types(), ["account_created"])

    def test_no_personal_data_is_stored(self):
        self._user_and_session(self._token())

        with self.s_maker() as s:
            rows = s.execute(AuthEvent.__table__.select()).mappings().all()
        stored = " ".join(str(v) for row in rows for v in row.values())
        for personal in ("jane@example.org", "Jane", "Doe", "jane"):
            self.assertNotIn(personal, stored)

    def test_logout_is_recorded_with_the_ended_session(self):
        account = self._user_and_session(self._token())

        response = self.client.post(
            f"{BASE}/logout",
            headers={
                **HEADERS,
                "Authorization": f"Bearer {self._token()}",
                "X-Session-Id": account["session_id"],
            },
        )

        self.assertEqual(response.status_code, 200)
        logouts = self._events(AuthEventType.LOGOUT)
        self.assertEqual(len(logouts), 1)
        self.assertEqual(str(logouts[0].inferred_user_id), account["user_id"])
        self.assertEqual(str(logouts[0].session_id), account["session_id"])
        self.assertEqual(logouts[0].identity_provider, "google")
        self.assertTrue(logouts[0].is_limited_access)
        self.assertIsNone(logouts[0].keycloak_session_id)

    def test_logout_without_token_or_with_invalid_token_records_nothing(self):
        account = self._user_and_session(self._token())
        session_headers = {**HEADERS, "X-Session-Id": account["session_id"]}

        invalid = self.client.post(
            f"{BASE}/logout",
            headers={**session_headers, "Authorization": "Bearer not.a.token"},
        )
        without_token = self.client.post(f"{BASE}/logout", headers=session_headers)

        self.assertEqual(invalid.status_code, 200)
        self.assertEqual(without_token.status_code, 200)
        self.assertEqual(self._events(AuthEventType.LOGOUT), [])

    def test_logout_of_an_unknown_account_records_nothing(self):
        response = self.client.post(
            f"{BASE}/logout",
            headers={**HEADERS, "Authorization": f"Bearer {self._token()}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._events(), [])

    @mock.patch(f"{SIGNUP}.send_signup_email", new_callable=mock.AsyncMock)
    @mock.patch(f"{SIGNUP}.create_user", new_callable=mock.AsyncMock)
    @mock.patch(f"{SIGNUP}.find_user_id_by_email", new_callable=mock.AsyncMock)
    def test_signup_of_a_guest_records_signup_and_new_account(
        self, find_mock, create_mock, email_mock
    ):
        find_mock.return_value = None
        create_mock.return_value = self.keycloak_id
        guest = self._user_and_session()

        response = self.client.post(
            f"{BASE}/signup",
            json={"email": "jane@example.org", "first_name": "Jane"},
            headers={**HEADERS, "X-Session-Id": guest["session_id"]},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(self._event_types(), ["account_created", "signup"])
        for event in self._events():
            self.assertEqual(str(event.inferred_user_id), guest["user_id"])
            self.assertEqual(str(event.session_id), guest["session_id"])
            # an email/password account
            self.assertIsNone(event.identity_provider)
            self.assertTrue(event.is_limited_access)

        # its first login is not a new account
        self._user_and_session(self._token(provider=None), guest["session_id"])
        self.assertEqual(self._event_types(), ["account_created", "login", "signup"])

    @mock.patch(f"{SIGNUP}.find_user_id_by_email", new_callable=mock.AsyncMock)
    def test_signup_of_a_registered_email_records_nothing(self, find_mock):
        find_mock.return_value = uuid.uuid4()

        response = self.client.post(
            f"{BASE}/signup", json={"email": "jane@example.org"}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(self._events(), [])

    @mock.patch.object(auth_events, "add_auth_event_sync")
    def test_db_failure_does_not_change_the_responses(self, add_mock):
        add_mock.side_effect = OperationalError("INSERT", {}, Exception())

        account = self._user_and_session(self._token())
        logout = self.client.post(
            f"{BASE}/logout",
            headers={
                **HEADERS,
                "Authorization": f"Bearer {self._token()}",
                "X-Session-Id": account["session_id"],
            },
        )

        # account_created, login, logout were attempted
        self.assertEqual(add_mock.call_count, 3)
        with self.s_maker() as s:
            user = s.get(InferredUser, uuid.UUID(account["user_id"]))
            self.assertEqual(user.keycloak_id, self.keycloak_id)
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(logout.json(), {"message": "logged out"})

    @mock.patch(f"{SIGNUP}.send_signup_email", new_callable=mock.AsyncMock)
    @mock.patch(f"{SIGNUP}.create_user", new_callable=mock.AsyncMock)
    @mock.patch(f"{SIGNUP}.find_user_id_by_email", new_callable=mock.AsyncMock)
    @mock.patch.object(auth_events, "add_auth_event_sync")
    def test_db_failure_does_not_change_the_signup_response(
        self, add_mock, find_mock, create_mock, email_mock
    ):
        add_mock.side_effect = OperationalError("INSERT", {}, Exception())
        find_mock.return_value = None
        create_mock.return_value = self.keycloak_id

        response = self.client.post(
            f"{BASE}/signup", json={"email": "jane@example.org"}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(add_mock.call_count, 2)
        email_mock.assert_awaited_once_with(self.keycloak_id)


if __name__ == "__main__":
    unittest.main()
