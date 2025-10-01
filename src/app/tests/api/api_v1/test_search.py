from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock
from unittest.mock import patch

from fastapi.testclient import TestClient
from qdrant_client.http import models

from src.app.core.config import settings
from src.app.models import collections
from src.app.models.search import EnhancedSearchQuery
from src.app.services.exceptions import (
    CollectionNotFoundError,
    LanguageNotSupportedError,
    ModelNotFoundError,
)
from src.app.services.search import SearchService, sort_slices_using_mmr
from src.main import app

client = TestClient(app)

search_pipeline_path = "src.app.services.search.SearchService"

mocked_collection = collections.Collection(
    lang="fr",
    model="model",
    name="collection_welearn_fr_model",
)
mocked_scored_points = [
    models.ScoredPoint(
        id="1",
        version=1,
        score=0.9,
        vector=[0.1, 0.2],
        payload={
            "document_corpus": "corpus",
            "document_desc": "desc",
            "document_details": {},
            "document_id": "1",
            "document_lang": "fr",
            "document_sdg": [1],
            "document_title": "title",
            "document_url": "url",
            "slice_content": "content",
            "slice_sdg": 1,
        },
    ),
    models.ScoredPoint(
        id="2",
        version=1,
        score=0.89,
        vector=[0.11, 0.21],
        payload={
            "document_corpus": "corpus",
            "document_desc": "desc",
            "document_details": {},
            "document_id": "1",
            "document_lang": "fr",
            "document_sdg": [1],
            "document_title": "title",
            "document_url": "url",
            "slice_content": "content",
            "slice_sdg": 1,
        },
    ),
    models.ScoredPoint(
        id="3",
        version=1,
        score=0.88,
        vector=[0.3, 0.4],
        payload={
            "document_corpus": "corpus",
            "document_desc": "desc",
            "document_details": {},
            "document_id": "1",
            "document_lang": "fr",
            "document_sdg": [1],
            "document_title": "title",
            "document_url": "url",
            "slice_content": "content",
            "slice_sdg": 1,
        },
    ),
]


long_query = "français with a very long sentence to test what you are saying and if the issue is the size of the string"  # noqa: E501


@patch("src.app.services.sql_db.session_maker")
@patch("src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True))
@patch(
    f"{search_pipeline_path}.get_collections",
    new=mock.AsyncMock(
        return_value=("collection_welearn_fr_model", "collection_en_model")
    ),
)
class SearchTests(IsolatedAsyncioTestCase):
    def test_search_items_no_query(self, *mocks):
        """Test search_items when no query is provided"""

        response = client.post(
            f"{settings.API_V1_STR}/search/collections/collection_welearn_fr_model",  # noqa: E501
            json={"nb_results": 10},
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 422)

    @patch(
        f"{search_pipeline_path}._get_model",
        new=mock.MagicMock(
            side_effect=ModelNotFoundError("Model not found", "MODEL_NOT_FOUND")
        ),
    )
    async def test_search_model_not_found(self, *mocks):
        with self.assertRaises(ModelNotFoundError):
            response = client.post(
                f"{settings.API_V1_STR}/search/collections/collection_welearn_fr_model?query=français&nb_results=10",  # noqa: E501
                headers={"X-API-Key": "test"},
            )

            self.assertEqual(response.status_code, 404)
            self.assertEqual(
                response.json(),
                {
                    "detail": {
                        "message": "Model not found",
                        "code": "MODEL_NOT_FOUND",
                    }
                },
            )

    @patch(
        f"{search_pipeline_path}.search_handler",
        new=mock.AsyncMock(return_value=mocked_scored_points),
    )
    async def test_search_items_success(self, *mocks):
        """Test successful search_items response"""

        response = client.post(
            f"{settings.API_V1_STR}/search/collections/collection_welearn_fr_model?query={long_query}&nb_results=10",
            headers={"X-API-Key": "test"},  # noqa: E501
        )

        self.assertEqual(response.status_code, 200)

    @patch(
        f"{search_pipeline_path}.search_handler",
        new=mock.AsyncMock(return_value=[]),
    )
    async def test_search_items_no_result(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/collections/collection_welearn_fr_model?query={long_query}&nb_results=10",
            headers={"X-API-Key": "test"},  # noqa: E501
        )

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.json(), [])

    @patch(
        f"{search_pipeline_path}.get_collection_by_language",
        new=mock.AsyncMock(
            side_effect=CollectionNotFoundError(
                "Collection not found", "COLL_NOT_FOUND"
            )
        ),
    )
    async def test_search_all_slices_no_collections(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/collections/collection_welearn_fr_model?query={long_query}&nb_results=10",
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            "Collection not found",
        )


@patch("src.app.services.sql_db.session_maker")
@patch("src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True))
class SearchTestsSlices(IsolatedAsyncioTestCase):
    async def test_search_all_slices_lang_not_supported(self, *mocks):
        with self.assertRaises(LanguageNotSupportedError):
            response = client.post(
                f"{settings.API_V1_STR}/search/by_slices",
                json={
                    "query": "pesquisa em portugues, mas longa para poder ser utilizada, ainda mais longa e com acentos éé e cedilhas çç"  # noqa: E501,
                },
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json(),
                {
                    "detail": {
                        "message": "Language not supported",
                        "code": "LANG_NOT_SUPPORTED",
                    }
                },
            )

    @patch(
        f"{search_pipeline_path}.get_collection_by_language",
        new=mock.AsyncMock(
            side_effect=CollectionNotFoundError(
                "Collection not found", "COLL_NOT_FOUND"
            )
        ),
    )
    async def test_search_all_slices_no_collections(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_slices?nb_results=10",
            json={
                "query": "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne"
            },  # noqa: E501
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            "Collection not found",
        )

    @patch(f"{search_pipeline_path}.search_handler", return_value=mocked_scored_points)
    async def test_search_all_slices_ok(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_slices",
            json={
                "query": "Comment est-ce que les gouvernements font pour suivre ces conseils et les mettre en place ?",
                "relevance_factor": 0.75,
            },
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)

    async def test_search_all_slices_no_query(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_slices",
            json={"query": ""},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json().get("detail")["message"],
            "Empty query",
        )

    @patch(
        f"{search_pipeline_path}.search_handler",
        return_value=[],
    )
    async def test_search_all_slices_no_result(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_slices?nb_results=10",
            json={
                "query": "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne"
            },
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 204)


@patch("src.app.services.sql_db.session_maker")
@patch("src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True))
class SearchTestsAll(IsolatedAsyncioTestCase):
    async def test_search_all_lang_not_supported(self, *mocks):
        with self.assertRaises(LanguageNotSupportedError):
            response = client.post(
                f"{settings.API_V1_STR}/search/by_document",
                json={
                    "query": "uma frase longa para fazer uma pesquisa el portugues e tes um erro porque nao é suportado"
                },  # noqa: E501
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json(),
                {
                    "detail": {
                        "message": "Language not supported",
                        "code": "LANG_NOT_SUPPORTED",
                    }
                },
            )

    @patch(
        f"{search_pipeline_path}.get_collection_by_language",
        new=mock.AsyncMock(
            side_effect=CollectionNotFoundError(
                "Collection not found", "COLL_NOT_FOUND"
            )
        ),
    )
    @patch(
        f"{search_pipeline_path}._get_info_from_collection_name",
        new=mock.MagicMock(return_value=mocked_collection),
    )
    async def test_search_all_no_collections(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_document?nb_results=10",
            json={
                "query": "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne"
            },
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(),
            "Collection not found",
        )

    @patch(f"{search_pipeline_path}.search_handler", return_value=[])
    async def test_search_all_no_result(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_document?nb_results=10",
            json={
                "query": "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne"
            },
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 204)

    async def test_search_all_no_query(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_document",
            json={"query": ""},
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json().get("detail")["message"],
            "Empty query",
        )


class TestSortSlicesUsingMMR(IsolatedAsyncioTestCase):
    async def test_sort_slices_using_mmr_default_theta(self, *mocks):
        sorted_points = sort_slices_using_mmr(mocked_scored_points)
        self.assertEqual(sorted_points, mocked_scored_points)

    async def test_sort_slices_using_mmr_custom_theta(self, *mocks):
        theta = 0.5
        sorted_points = sort_slices_using_mmr(mocked_scored_points, theta)
        self.assertEqual(
            sorted_points,
            [mocked_scored_points[0], mocked_scored_points[2], mocked_scored_points[1]],
        )


@patch("src.app.services.sql_db.session_maker")
@patch("src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True))
class SearchTestsMultiInput(IsolatedAsyncioTestCase):
    async def test_search_multi_lang_not_supported(self, *mocks):
        with self.assertRaises(LanguageNotSupportedError):
            response = client.post(
                f"{settings.API_V1_STR}/search/multiple_by_slices",
                json={
                    "query": [
                        "uma frase longa para fazer uma pesquisa el portugues e tes um erro porque nao é suportado",
                        "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne",
                    ]  # noqa: E501
                },  # noqa: E501
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json(),
                {
                    "detail": {
                        "message": "Language not supported",
                        "code": "LANG_NOT_SUPPORTED",
                    }
                },
            )

    @patch(
        f"{search_pipeline_path}.search_handler",
        return_value=[],
    )
    async def test_search_multi_no_result(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/multiple_by_slices?nb_results=10",
            json={
                "query": [
                    "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne",
                    "another long sentence to test the search in english and see what happens",
                ]
            },
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 204)


@patch("src.app.api.api_v1.endpoints.search.session_maker")
@patch("src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True))
class DocumentsByIdsTests(IsolatedAsyncioTestCase):
    async def test_documents_by_ids_empty(self, session_maker_mock, *mocks):
        session = session_maker_mock.return_value.__enter__.return_value
        exec_docs = mock.MagicMock()
        exec_docs.all.return_value = []
        exec_corpora = mock.MagicMock()
        exec_corpora.all.return_value = []
        exec_slices = mock.MagicMock()
        exec_slices.all.return_value = []
        exec_sdgs = mock.MagicMock()
        exec_sdgs.all.return_value = []

        session.execute.side_effect = [exec_docs, exec_corpora, exec_slices, exec_sdgs]

        response = client.post(
            f"{settings.API_V1_STR}/search/documents/by_ids",
            json=[],
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    async def test_documents_by_ids_single_doc(self, session_maker_mock, *mocks):
        session = session_maker_mock.return_value.__enter__.return_value

        doc_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        corpus_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        docs_row = SimpleNamespace(
            title="Title",
            url="https://example.com",
            corpus_id=corpus_id,
            id=doc_id,
            description="Desc",
            details={"k": "v"},
        )

        # 1) documents .all()
        exec_docs = mock.MagicMock()
        exec_docs.all.return_value = [docs_row]
        # 2) corpora .all()
        exec_corpora = mock.MagicMock()
        exec_corpora.all.return_value = [
            SimpleNamespace(id=corpus_id, source_name="Corpus")
        ]
        # 3) slices .all()
        slice1 = "11111111-1111-1111-1111-111111111111"
        slice2 = "22222222-2222-2222-2222-222222222222"
        exec_slices = mock.MagicMock()
        exec_slices.all.return_value = [
            SimpleNamespace(id=slice1, document_id=doc_id),
            SimpleNamespace(id=slice2, document_id=doc_id),
        ]
        # 4) sdgs .all()
        exec_sdgs = mock.MagicMock()
        exec_sdgs.all.return_value = [
            SimpleNamespace(sdg_number=1, slice_id=slice1),
            SimpleNamespace(sdg_number=3, slice_id=slice2),
            SimpleNamespace(sdg_number=1, slice_id=slice2),
        ]

        session.execute.side_effect = [exec_docs, exec_corpora, exec_slices, exec_sdgs]

        response = client.post(
            f"{settings.API_V1_STR}/search/documents/by_ids",
            json=[doc_id],
            headers={"X-API-Key": "test"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        payload = body[0]
        self.assertEqual(payload["document_id"], doc_id)
        self.assertEqual(payload["document_title"], "Title")
        self.assertEqual(payload["document_url"], "https://example.com")
        self.assertEqual(payload["document_desc"], "Desc")
        self.assertEqual(payload["document_details"], {"k": "v"})
        self.assertEqual(payload["document_corpus"], "Corpus")
        self.assertEqual(payload["document_sdg"], [1, 3])
        self.assertEqual(payload["slice_content"], "")
        self.assertIsNone(payload["slice_sdg"])

    async def test_documents_by_ids_corpus_missing(self, session_maker_mock, *mocks):
        session = session_maker_mock.return_value.__enter__.return_value

        doc_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        corpus_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        docs_row = SimpleNamespace(
            title="Title",
            url="https://example.com",
            corpus_id=corpus_id,
            id=doc_id,
            description="Desc",
            details={},
        )

        exec_docs = mock.MagicMock()
        exec_docs.all.return_value = [docs_row]
        exec_corpora = mock.MagicMock()
        exec_corpora.all.return_value = []
        exec_slices = mock.MagicMock()
        exec_slices.all.return_value = []
        exec_sdgs = mock.MagicMock()
        exec_sdgs.all.return_value = []
        session.execute.side_effect = [exec_docs, exec_corpora, exec_slices, exec_sdgs]

        response = client.post(
            f"{settings.API_V1_STR}/search/documents/by_ids",
            json=[doc_id],
            headers={"X-API-Key": "test"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["document_corpus"], "")

    async def test_search_multi_single_query(self, *mocks):
        with mock.patch(
            "src.app.api.api_v1.endpoints.search.search_multi_inputs",
        ) as search_multi, mock.patch.object(
            SearchService, "search_handler", return_value=mocked_scored_points
        ) as search_handler:
            client.post(
                f"{settings.API_V1_STR}/search/multiple_by_slices?nb_results=10",
                json={
                    "query": long_query,
                },
                headers={"X-API-Key": "test"},
            )
            search_multi.assert_called_once_with(
                qp=EnhancedSearchQuery(
                    query=[long_query],
                    sdg_filter=None,
                    corpora=None,
                    subject=None,
                    nb_results=10,
                    influence_factor=2.0,
                    relevance_factor=1.0,
                ),
                callback_function=search_handler,  # noqa: E501
            )
