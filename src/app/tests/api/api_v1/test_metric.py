import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.main import app

client = TestClient(app)

MOCK_RESULT = [
    (
        mock.Mock(source_name="corpus1", main_url="http://example.com/corpus1"),
        mock.Mock(count=5),
        mock.Mock(count=10),
    ),
    (
        mock.Mock(source_name="corpus2", main_url="http://example.org/corpus2"),
        mock.Mock(count=0),
        mock.Mock(count=0),
    ),
]


@mock.patch(
    "src.app.services.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class TestMetricEndpoint(unittest.IsolatedAsyncioTestCase):
    @mock.patch("src.app.api.api_v1.endpoints.metric.get_document_qty_table_info_sync")
    async def test_nb_docs_info_per_corpus_ok(self, mock_get_info):
        mock_get_info.return_value = MOCK_RESULT
        response = client.get(
            f"{settings.API_V1_STR}/metric/nb_docs_info_per_corpus",
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["corpus"], "corpus1")
        self.assertEqual(data[0]["url"], "http://example.com/corpus1")
        self.assertEqual(data[0]["qty_total"], 10)
        self.assertEqual(data[0]["qty_in_qdrant"], 5)
        self.assertEqual(data[1]["corpus"], "corpus2")
        self.assertEqual(data[1]["qty_total"], 0)

    @mock.patch("src.app.api.api_v1.endpoints.metric.get_document_qty_table_info_sync")
    async def test_nb_docs_info_per_corpus_empty(self, mock_get_info):
        mock_get_info.return_value = []
        response = client.get(
            f"{settings.API_V1_STR}/metric/nb_docs_info_per_corpus",
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    @mock.patch("src.app.api.api_v1.endpoints.metric.get_document_qty_table_info_sync")
    async def test_nb_docs_info_per_corpus_none(self, mock_get_info):
        mock_get_info.return_value = None
        response = client.get(
            f"{settings.API_V1_STR}/metric/nb_docs_info_per_corpus",
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    @mock.patch("src.app.api.api_v1.endpoints.metric.get_document_qty_table_info_sync")
    async def test_nb_docs_info_per_corpus_no_content(self, mock_get_info):
        # Cas où certains champs sont None ou manquants
        partial_result = [
            (
                mock.Mock(source_name=None, main_url=None),
                mock.Mock(count=None),
                mock.Mock(count=None),
            ),
        ]
        mock_get_info.return_value = partial_result
        response = client.get(
            f"{settings.API_V1_STR}/metric/nb_docs_info_per_corpus",
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 500)

    @mock.patch("src.app.api.api_v1.endpoints.metric.get_document_qty_table_info_sync")
    async def test_nb_docs_info_per_corpus_partial(self, mock_get_info):
        # Cas où certains champs sont None ou manquants
        partial_result = [
            (
                mock.Mock(source_name=None, main_url=None),
                mock.Mock(count=None),
                mock.Mock(count=None),
            ),
            (
                mock.Mock(source_name="corpus1", main_url="http://example.com/corpus1"),
                mock.Mock(count=5),
                mock.Mock(count=10),
            ),
        ]
        mock_get_info.return_value = partial_result
        response = client.get(
            f"{settings.API_V1_STR}/metric/nb_docs_info_per_corpus",
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 206)

        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["corpus"], "corpus1")
        self.assertEqual(data[0]["url"], "http://example.com/corpus1")
        self.assertEqual(data[0]["qty_total"], 10)
        self.assertEqual(data[0]["qty_in_qdrant"], 5)
