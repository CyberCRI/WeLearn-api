import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.main import app

client = TestClient(app)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class BibliographyApiTests(unittest.IsolatedAsyncioTestCase):
    @mock.patch("src.app.bibliography.api.bibliography.welearn_document_to_ris")
    @mock.patch("src.app.bibliography.api.bibliography.get_documents_by_ids")
    async def test_export_bibliography_success(
        self, get_documents_by_ids_mock, welearn_document_to_ris_mock, *mocks
    ):
        doc1 = mock.sentinel.doc1
        doc2 = mock.sentinel.doc2
        get_documents_by_ids_mock.return_value = [doc1, doc2]
        welearn_document_to_ris_mock.side_effect = [
            "TY  - JFULL\nER  - ",
            "TY  - BOOK\nER  - ",
        ]

        payload = {
            "documents_ids": [
                "12345678-1234-5678-1234-567812345678",
                "87654321-4321-8765-4321-876543218765",
            ]
        }
        response = client.post(
            f"{settings.API_V1_STR}/bibliography/export_bibliography",
            json=payload,
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "TY  - JFULL\nER  - \nTY  - BOOK\nER  - \n")
        get_documents_by_ids_mock.assert_called_once_with(payload["documents_ids"])
        welearn_document_to_ris_mock.assert_has_calls(
            [mock.call(doc1), mock.call(doc2)]
        )

    @mock.patch("src.app.bibliography.api.bibliography.welearn_document_to_ris")
    @mock.patch("src.app.bibliography.api.bibliography.get_documents_by_ids")
    async def test_export_bibliography_empty_documents(
        self, get_documents_by_ids_mock, welearn_document_to_ris_mock, *mocks
    ):
        get_documents_by_ids_mock.return_value = []

        response = client.post(
            f"{settings.API_V1_STR}/bibliography/export_bibliography",
            json={"documents_ids": ["12345678-1234-5678-1234-567812345678"]},
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "")
        welearn_document_to_ris_mock.assert_not_called()

    def test_export_bibliography_invalid_payload(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/bibliography/export_bibliography",
            json={"documents_ids": ["not-a-uuid"]},
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 422)
