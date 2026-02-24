import io
from unittest import IsolatedAsyncioTestCase, mock

from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.main import app

# client = TestClient(app)


@mock.patch("src.app.services.sql_db.sql_service.session_maker")
@mock.patch(
    "src.app.services.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class TutorTests(IsolatedAsyncioTestCase):
    def test_tutor_no_files(self, *mocks):
        with TestClient(app) as client:
            response = client.post(
                f"{settings.API_V1_STR}/tutor/search",
                files={},
                headers={"x-API-Key": "test"},
            )
            assert response.status_code == 422

    def test_tutor_empty_file(self, *mocks):
        file = io.BytesIO(b"")
        with TestClient(app) as client:
            response = client.post(
                f"{settings.API_V1_STR}/tutor/search",
                files={"files": ("test.txt", file)},
                headers={"x-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 400)

    def test_tutor_file(self, *mocks):
        file = io.BytesIO(b"this is a test file")
        with TestClient(app) as client:
            response = client.post(
                f"{settings.API_V1_STR}/tutor/search",
                files={"files": ("test.txt", file)},
                headers={"x-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 204)
