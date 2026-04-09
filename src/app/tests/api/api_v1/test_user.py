import unittest
import uuid
from unittest import mock
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from welearn_database.data.models import InferredUser, Session

from src.app.core.config import settings
from src.app.shared.domain.exceptions import UserNotFoundError
from src.main import app

client = TestClient(app)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class UserApiTests(unittest.IsolatedAsyncioTestCase):

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_user_when_not_exists(self, session_maker_mock, *mocks):
        """Si user_id non fourni, crée un nouvel utilisateur"""
        session = MagicMock()
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/user",
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("user_id", response.json())
        session.add.assert_called_once()
        session.commit.assert_called_once()
        user_in_db: InferredUser = session.add.call_args[0][0]
        self.assertEqual(user_in_db.id, response.json()["user_id"])
        self.assertIsNone(user_in_db.origin_referrer)

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_user_when_not_exists_with_referer(
        self, session_maker_mock, *mocks
    ):
        """Si user_id non fourni, crée un nouvel utilisateur"""
        session = MagicMock()
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/user?referer=test_referer",
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("user_id", response.json())
        session.add.assert_called_once()
        session.commit.assert_called_once()
        user_in_db: InferredUser = session.add.call_args[0][0]
        self.assertEqual(user_in_db.id, response.json()["user_id"])
        self.assertEqual(user_in_db.origin_referrer, "test_referer")

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_user_when_already_exists(self, session_maker_mock, *mocks):
        """Si user_id fourni et trouvé, retourne le même id sans créer"""
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
        session.add.assert_not_called()
        session.commit.assert_not_called()

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_user_when_already_exists_with_referer(
        self, session_maker_mock, *mocks
    ):
        """Si user_id fourni et trouvé, retourne le même id sans créer"""
        session = MagicMock()
        session.execute.return_value.first.return_value = MagicMock(
            id="cfc8072c-a055-442a-9878-b5a73d9141b2"
        )
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/user",
            params={
                "user_id": "cfc8072c-a055-442a-9878-b5a73d9141b2",
                "referer": "test_referer",
            },
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"user_id": "cfc8072c-a055-442a-9878-b5a73d9141b2"}
        )
        session.add.assert_not_called()
        session.commit.assert_not_called()

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_user_handles_exception(self, session_maker_mock, *mocks):
        """Simule une erreur DB et vérifie que l’API renvoie 500"""
        session = MagicMock()
        session.add.side_effect = Exception("db error")
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/user",
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("db error", response.json()["detail"])

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_session_user_not_found(self, session_maker_mock, *mocks):
        """User inexistant -> 404"""
        session = MagicMock()
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "bdb62bb2-1fe5-4d14-92fd-60a041355aea"
        response = client.post(
            f"{settings.API_V1_STR}/user/session",
            params={"user_id": user_id},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_session_existing_valid_session(
        self, session_maker_mock, *mocks
    ):
        """User et session existants -> retourne la session existante"""
        session = MagicMock()
        session.execute.return_value.first.side_effect = [
            MagicMock(id="cfc8072c-a055-442a-9878-b5a73d9141b2"),  # check user exists
            MagicMock(
                id="bdb62bb2-1fe5-4d14-92fd-60a041355aea"
            ),  # check session exists
        ]
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/session",
            params={
                "user_id": "cfc8072c-a055-442a-9878-b5a73d9141b2",
                "session_id": "bdb62bb2-1fe5-4d14-92fd-60a041355aea",
            },
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"session_id": "bdb62bb2-1fe5-4d14-92fd-60a041355aea"}
        )

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_session_create_new_when_not_found(
        self, session_maker_mock, *mocks
    ):
        """User existant mais session non trouvée -> crée nouvelle session"""
        session = MagicMock()
        session.execute.return_value.first.side_effect = [MagicMock(id="user-1"), None]
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/session",
            params={"user_id": "cfc8072c-a055-442a-9878-b5a73d9141b2"},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("session_id", response.json())
        session.add.assert_called_once()
        session.commit.assert_called_once()

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_create_session_create_new_when_not_found_with_referer(
        self, session_maker_mock, *mocks
    ):
        """User existant mais session non trouvée -> crée nouvelle session"""
        session = MagicMock()
        session.execute.return_value.first.side_effect = [MagicMock(id="user-1"), None]
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.post(
            f"{settings.API_V1_STR}/user/session",
            params={
                "user_id": "cfc8072c-a055-442a-9878-b5a73d9141b2",
                "referer": "test_referer",
            },
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("session_id", response.json())
        session.add.assert_called_once()
        session.commit.assert_called_once()

        session_in_db: Session = session.add.call_args[0][0]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(session_in_db.origin_referrer, "test_referer")

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_get_user_bookmarks_user_not_found(self, session_maker_mock, *mocks):
        """Bookmarks pour user inexistant -> 404"""
        session = MagicMock()
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        user_id = "11111111-1111-1111-1111-111111111111"
        response = client.get(
            f"{settings.API_V1_STR}/user/:user_id/bookmarks",
            params={"user_id": user_id},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 404)

    @mock.patch("src.app.services.sql_db.queries_user.session_maker")
    async def test_get_user_bookmarks_success_empty(self, session_maker_mock, *mocks):
        """Bookmarks existants -> liste vide si pas de bookmarks"""
        session = MagicMock()
        session.execute.return_value.first.return_value = MagicMock(id="user-1")
        session.execute.return_value.all.return_value = []
        session_maker_mock.return_value.__enter__.return_value = session

        response = client.get(
            f"{settings.API_V1_STR}/user/bookmarks",
            headers={"X-API-Key": "test"},
            cookies={"x-session-id": "bdb62bb2-1fe5-4d14-92fd-60a041355aea"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"bookmarks": []})

    @mock.patch("src.app.user.api.router.run_in_threadpool")
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    async def test_add_user_bookmark_success(
        self, resolve_user_and_session_mock, run_in_threadpool_mock, *mocks
    ):
        """Ajout d'un bookmark - mocks only what is needed"""
        # Mock resolve_user_and_session to return user_id and session_id
        user_id = uuid.UUID("cfc8072c-a055-442a-9878-b5a73d9141b2")
        session_id = uuid.UUID("bdb62bb2-1fe5-4d14-92fd-60a041355aea")
        resolve_user_and_session_mock.return_value = (user_id, session_id)

        # Mock run_in_threadpool to simulate DB add_user_bookmark_sync
        document_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        run_in_threadpool_mock.return_value = document_id

        response = client.post(
            f"{settings.API_V1_STR}/user/bookmarks/:document_id",
            params={"document_id": document_id},
            headers={"X-API-Key": "test"},
            cookies={"x-session-id": str(session_id)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"added": document_id})
        resolve_user_and_session_mock.assert_called_once()
        run_in_threadpool_mock.assert_called_once()

    @mock.patch("src.app.user.api.router.run_in_threadpool")
    @mock.patch("src.app.user.api.router.resolve_user_and_session")
    async def test_add_user_bookmark_user_not_found(
        self, resolve_user_and_session_mock, run_in_threadpool_mock, *mocks
    ):
        """Ajout d'un bookmark - mocks only what is needed"""
        # Mock resolve_user_and_session to return user_id and session_id
        user_id = uuid.UUID("cfc8072c-a055-442a-9878-b5a73d9141b2")
        session_id = uuid.UUID("bdb62bb2-1fe5-4d14-92fd-60a041355aea")

        # mock error for user not found
        resolve_user_and_session_mock.side_effect = UserNotFoundError("User not found")
        resolve_user_and_session_mock.return_value = (user_id, session_id)

        # Mock run_in_threadpool to simulate DB add_user_bookmark_sync
        document_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        run_in_threadpool_mock.return_value = document_id

        response = client.post(
            f"{settings.API_V1_STR}/user/bookmarks/:document_id",
            params={"document_id": document_id},
            headers={"X-API-Key": "test"},
            cookies={"x-session-id": str(session_id)},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(), {"detail": "('User not found', 'USER_NOT_FOUND')"}
        )
        resolve_user_and_session_mock.assert_called_once()
        run_in_threadpool_mock.assert_not_called()
