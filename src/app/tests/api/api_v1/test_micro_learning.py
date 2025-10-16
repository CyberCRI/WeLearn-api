import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.app.models.collections import Collection
from welearn_database.data.models import ContextDocument
from src.main import app

client = TestClient(app)


class AsyncMock(mock.MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


@mock.patch(
    "src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True)
)
class MicroLearningTests(unittest.IsolatedAsyncioTestCase):
    @mock.patch(
        "src.app.api.api_v1.endpoints.micro_learning.collection_and_model_id_according_lang",
        new_callable=AsyncMock,
    )
    @mock.patch("src.app.api.api_v1.endpoints.micro_learning.get_context_documents")
    @mock.patch("src.app.api.api_v1.endpoints.micro_learning.get_subject")
    @mock.patch("src.app.services.search.SearchService.search")
    @mock.patch("src.app.api.api_v1.endpoints.micro_learning.convert_embedding_bytes")
    async def test_get_full_journey(
        self,
        mock_convert_embedding,
        mock_search,
        mock_get_subject,
        mock_get_context_docs,
        mock_collection_and_model_id_according_lang,
    ):
        mock_collection_and_model_id_according_lang.return_value = (
            Collection(name="test_collection", lang="en", model="test_model"),
            "model_id",
        )
        # Mock data
        mock_get_context_docs.return_value = [
            ContextDocument(
                id="test_id",
                title="Test Title",
                full_content="Test Content",
                embedding=b"test_embedding",
                context_type="introduction",
            ),
            ContextDocument(
                id="test_id2",
                title="Test Title target 1",
                full_content="Test Content",
                embedding=b"test_embedding",
                context_type="target",
            ),
            ContextDocument(
                id="test_id3",
                title="Test Title target 2",
                full_content="Test Content",
                embedding=b"test_embedding",
                context_type="target",
            ),
        ]
        mock_get_subject.return_value = ContextDocument(
            id="subject_id",
            title="Test Subject",
            embedding=b"subject_embedding",
            context_type="subject",
        )
        mock_convert_embedding.return_value = [0.1, 0.2, 0.3]
        mock_search.return_value = [{"id": "doc1", "title": "Doc 1"}]

        # API call
        response = client.get(
            f"{settings.API_V1_STR}/micro_learning/full_journey",
            params={"lang": "en", "sdg": 1, "subject": "Test Subject"},
            headers={"X-API-Key": "test"},
        )
        # Assertions
        self.assertIn("introduction", response.json())
        self.assertEqual(len(response.json()["introduction"]), 1)
        self.assertEqual(response.json()["introduction"][0]["title"], "Test Title")
        self.assertEqual(
            response.json()["introduction"][0]["documents"][0]["title"], "Doc 1"
        )
        self.assertIn("target", response.json())
        self.assertEqual(len(response.json()["target"]), 2)

    @mock.patch(
        "src.app.api.api_v1.endpoints.micro_learning.collection_and_model_id_according_lang",
        new_callable=AsyncMock,
    )
    @mock.patch("src.app.api.api_v1.endpoints.micro_learning.get_subjects")
    async def test_get_subject_list(
        self, mock_get_subjects, mock_collection_and_model_id_according_lang
    ):
        mock_get_subjects.return_value = [
            ContextDocument(
                id="subject_id",
                title="subject0",
                embedding=b"subject_embedding",
                context_type="subject",
            ),
            ContextDocument(
                id="subject_id2",
                title="subject1",
                embedding=b"subject_embedding",
                context_type="subject",
            ),
        ]
        mock_collection_and_model_id_according_lang.return_value = (None, "model_id")

        # API call
        response = client.get(
            f"{settings.API_V1_STR}/micro_learning/subject_list",
            headers={"X-API-Key": "test"},
        )

        ret = response.json()

        self.assertListEqual(["subject0", "subject1"], ret)
