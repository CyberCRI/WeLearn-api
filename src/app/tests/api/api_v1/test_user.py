import unittest
from unittest import mock
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.main import app

client = TestClient(app)


@mock.patch(
    "src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True)
)
class UserApiTests(unittest.IsolatedAsyncioTestCase):
    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_create_user_when_not_exists(self, session_maker_mock, *mocks):
        # user_id not provided -> should create new user
        session = MagicMock()
        session_maker_mock.return_value.__enter__.return_value = session
        # No select is made when user_id is None; just ensure add/commit are called
        response = client.post(
            f"{settings.API_V1_STR}/user/user", headers={"X-API-Key": "test"}
        )
        self.assertEqual(response.status_code, 200)
        session.add.assert_called()
        session.commit.assert_called()
        self.assertIn("user_id", response.json())

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_create_user_when_already_exists(self, session_maker_mock, *mocks):
        # user_id provided and found -> should not create, returns same id
        session = MagicMock()
        session.execute.return_value.first.return_value = MagicMock(
            id="cfc8072c-a055-442a-9878-b5a73d9141b2"
        )
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/user",
            params={"user_id": "cfc8072c-a055-442a-9878-b5a73d9141b2"},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"user_id": "cfc8072c-a055-442a-9878-b5a73d9141b2"}
        )

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_create_user_handles_exception(self, session_maker_mock, *mocks):
        # simulate exception during DB operation
        session = MagicMock()
        session.add.side_effect = Exception("db error")
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/user", headers={"X-API-Key": "test"}
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("Error creating user", response.json()["detail"])

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_create_session_user_not_found(self, session_maker_mock, *mocks):
        # user_id not found -> 404
        session = MagicMock()
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/session",
            params={"user_id": "bdb62bb2-1fe5-4d14-92fd-60a041355aea"},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_create_session_existing_valid_session(
        self, session_maker_mock, *mocks
    ):
        # user exists and a valid session exists -> return that session id
        session = MagicMock()
        # First call: check user exists
        session.execute.return_value.first.side_effect = [
            MagicMock(id="19f11fa7-87ef-40af-aa61-96a099bd04be"),
            MagicMock(id="8178c3c4-9379-4997-a6e6-f1ccea7a30a9"),
        ]
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/session",
            params={
                "user_id": "19f11fa7-87ef-40af-aa61-96a099bd04be",
                "session_id": "8178c3c4-9379-4997-a6e6-f1ccea7a30a9",
            },
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"session_id": "8178c3c4-9379-4997-a6e6-f1ccea7a30a9"}
        )

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_create_session_create_new_when_not_found(
        self, session_maker_mock, *mocks
    ):
        # user exists but no valid session -> create new session
        session = MagicMock()
        # First call: user exists; Second call: session not found
        session.execute.return_value.first.side_effect = [MagicMock(id="user-1"), None]
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/session",
            params={"user_id": "ca592fd6-15af-4272-8503-49347e8a2c5b"},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("session_id", response.json())
        session.add.assert_called()
        session.commit.assert_called()

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_create_session_handles_exception(self, session_maker_mock, *mocks):
        # simulate exception during session creation path
        session = MagicMock()
        # User exists
        session.execute.return_value.first.side_effect = [MagicMock(id="user-1"), None]
        session.add.side_effect = Exception("db error")
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/session",
            params={"user_id": "8bb64641-7196-4979-8d71-9d87898640b9"},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("Error creating session", response.json()["detail"])
