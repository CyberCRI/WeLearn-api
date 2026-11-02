import unittest
import uuid
from unittest.mock import AsyncMock, patch

from src.app.shared.domain.exceptions import SessionNotFoundError
from src.app.user.infra.keycloak import KeycloakUnavailableError
from src.app.user.infra.keycloak_admin import KeycloakUserExistsError
from src.app.user.services.signup import sign_up_user

SIGNUP = "src.app.user.services.signup"


@patch(f"{SIGNUP}.send_signup_email", new_callable=AsyncMock)
@patch(f"{SIGNUP}.get_or_create_user_from_keycloak_sync")
@patch(f"{SIGNUP}.get_user_from_session_id")
@patch(f"{SIGNUP}.create_user", new_callable=AsyncMock)
@patch(f"{SIGNUP}.find_user_id_by_email", new_callable=AsyncMock)
class TestSignUpUser(unittest.IsolatedAsyncioTestCase):
    async def test_email_already_has_account(
        self, find_mock, create_mock, session_mock, get_or_create_mock, email_mock
    ):
        find_mock.return_value = uuid.uuid4()

        with self.assertRaises(KeycloakUserExistsError):
            await sign_up_user("jane@example.org", None, None, uuid.uuid4(), None)

        create_mock.assert_not_called()
        get_or_create_mock.assert_not_called()
        email_mock.assert_not_called()

    async def test_new_user_without_session(
        self, find_mock, create_mock, session_mock, get_or_create_mock, email_mock
    ):
        keycloak_id, user_id = uuid.uuid4(), uuid.uuid4()
        find_mock.return_value = None
        create_mock.return_value = keycloak_id
        get_or_create_mock.return_value = user_id

        result = await sign_up_user("jane@example.org", "Jane", "Doe", None, "welearn")

        self.assertEqual(result, user_id)
        create_mock.assert_awaited_once_with("jane@example.org", "Jane", "Doe")
        session_mock.assert_not_called()
        get_or_create_mock.assert_called_once_with(keycloak_id, None, "welearn")
        email_mock.assert_awaited_once_with(keycloak_id)

    async def test_links_anonymous_user_from_session(
        self, find_mock, create_mock, session_mock, get_or_create_mock, email_mock
    ):
        keycloak_id, anonymous_user_id = uuid.uuid4(), uuid.uuid4()
        session_uuid = uuid.uuid4()
        find_mock.return_value = None
        create_mock.return_value = keycloak_id
        session_mock.return_value = anonymous_user_id
        get_or_create_mock.return_value = anonymous_user_id

        result = await sign_up_user("jane@example.org", None, None, session_uuid, None)

        self.assertEqual(result, anonymous_user_id)
        session_mock.assert_called_once_with(session_uuid)
        get_or_create_mock.assert_called_once_with(keycloak_id, anonymous_user_id, None)

    async def test_unknown_session_is_ignored(
        self, find_mock, create_mock, session_mock, get_or_create_mock, email_mock
    ):
        keycloak_id = uuid.uuid4()
        find_mock.return_value = None
        create_mock.return_value = keycloak_id
        session_mock.side_effect = SessionNotFoundError("Session not found")
        get_or_create_mock.return_value = uuid.uuid4()

        await sign_up_user("jane@example.org", None, None, uuid.uuid4(), None)

        get_or_create_mock.assert_called_once_with(keycloak_id, None, None)

    async def test_email_failure_still_succeeds(
        self, find_mock, create_mock, session_mock, get_or_create_mock, email_mock
    ):
        user_id = uuid.uuid4()
        find_mock.return_value = None
        create_mock.return_value = uuid.uuid4()
        get_or_create_mock.return_value = user_id
        email_mock.side_effect = KeycloakUnavailableError("smtp down")

        result = await sign_up_user("jane@example.org", None, None, None, None)

        self.assertEqual(result, user_id)


if __name__ == "__main__":
    unittest.main()
