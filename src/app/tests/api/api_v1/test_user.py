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

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_get_user_bookmarks_user_not_found(self, session_maker_mock, *mocks):
        session = MagicMock()
        # First DB call checks user existence -> None
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "11111111-1111-1111-1111-111111111111"
        response = client.get(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_get_user_bookmarks_success_empty(self, session_maker_mock, *mocks):
        session = MagicMock()
        # First call: user exists; Second call: bookmarks list -> []
        session.execute.return_value.first.return_value = [MagicMock(id="user-1"), None]
        # When calling .all() for bookmarks, return an empty list to avoid serialization issues
        session.execute.return_value.all.return_value = []
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "22222222-2222-2222-2222-222222222222"
        response = client.get(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("bookmarks", body)
        self.assertIsInstance(body["bookmarks"], list)
        self.assertEqual(len(body["bookmarks"]), 0)

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_delete_user_bookmarks_user_not_found(
        self, session_maker_mock, *mocks
    ):
        session = MagicMock()
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "33333333-3333-3333-3333-333333333333"
        response = client.delete(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_delete_user_bookmarks_success(self, session_maker_mock, *mocks):
        session = MagicMock()
        # user exists
        session.execute.return_value.first.return_value = MagicMock(id="user-1")
        # chain: query().filter().delete() -> 3
        session.query.return_value.filter.return_value.delete.return_value = 3
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "44444444-4444-4444-4444-444444444444"
        response = client.delete(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"deleted": 3})
        session.commit.assert_called()

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_delete_user_bookmark_user_not_found(
        self, session_maker_mock, *mocks
    ):
        session = MagicMock()
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "55555555-5555-5555-5555-555555555555"
        document_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        response = client.delete(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks/:document_id",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id, "document_id": document_id},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_delete_user_bookmark_not_found(self, session_maker_mock, *mocks):
        session = MagicMock()
        # user exists
        session.execute.return_value.first.side_effect = [MagicMock(id="user-1"), None]
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "66666666-6666-6666-6666-666666666666"
        document_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        response = client.delete(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks/:document_id",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id, "document_id": document_id},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_delete_user_bookmark_success(self, session_maker_mock, *mocks):
        session = MagicMock()
        # user exists then bookmark exists
        bookmark_obj = MagicMock()
        session.execute.return_value.first.side_effect = [
            MagicMock(id="user-1"),
            (bookmark_obj,),
        ]
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "77777777-7777-7777-7777-777777777777"
        document_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        response = client.delete(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks/:document_id",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id, "document_id": document_id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"deleted": document_id})
        session.delete.assert_called_with(bookmark_obj)
        session.commit.assert_called()

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_add_user_bookmark_user_not_found(self, session_maker_mock, *mocks):
        session = MagicMock()
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "88888888-8888-8888-8888-888888888888"
        document_id = "dddddddd-dddd-dddd-dddd-dddddddddddd"
        response = client.post(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks/:document_id",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id, "document_id": document_id},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_add_user_bookmark_already_exists(self, session_maker_mock, *mocks):
        session = MagicMock()
        # user exists, bookmark exists
        session.execute.return_value.first.side_effect = [
            MagicMock(id="user-1"),
            (MagicMock(),),
        ]
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "99999999-9999-9999-9999-999999999999"
        document_id = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
        response = client.post(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks/:document_id",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id, "document_id": document_id},
        )
        self.assertEqual(response.status_code, 400)

    @mock.patch("src.app.api.api_v1.endpoints.user.session_maker")
    async def test_add_user_bookmark_success(self, session_maker_mock, *mocks):
        session = MagicMock()
        # user exists, bookmark not found
        session.execute.return_value.first.side_effect = [MagicMock(id="user-1"), None]
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        document_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        response = client.post(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks/:document_id",
            headers={"X-API-Key": "test"},
            params={"user_id": user_id, "document_id": document_id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"added": document_id})
        session.add.assert_called()
        session.commit.assert_called()
