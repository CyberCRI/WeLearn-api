from unittest import IsolatedAsyncioTestCase, mock
from unittest.mock import patch

from fastapi.testclient import TestClient
from qdrant_client.http import models

from src.app.core.config import settings
from src.app.models import collections, documents
from src.app.services.exceptions import (
    CollectionNotFoundError,
    LanguageNotSupportedError,
    ModelNotFoundError,
    NoResultsError,
)
from src.app.services.search import sort_slices_using_mmr
from src.main import app

client = TestClient(app)

search_pipeline_path = "src.app.services.search.SearchService"
parallel_search_path = "src.app.api.api_v1.endpoints.search.parallel_search"

mocked_collection = collections.Collection(
    name="collection",
    lang="fr",
    model="model",
    alias="collection_fr_model",
)
mocked_scored_points = [
    models.ScoredPoint(
        id="1",
        version=1,
        score=0.9,
        vector=[0.1, 0.2],
    ),
    models.ScoredPoint(
        id="2",
        version=1,
        score=0.89,
        vector=[0.11, 0.21],
    ),
    models.ScoredPoint(
        id="3",
        version=1,
        score=0.88,
        vector=[0.3, 0.4],
    ),
]


mocked_document = documents.Document(
    score=0.9,
    payload=documents.DocumentPayloadModel(
        document_corpus="corpus",
        document_desc="desc",
        document_details={},
        document_id="1",
        document_lang="fr",
        document_sdg=[1],
        document_title="title",
        document_url="url",
        slice_content="content",
        slice_sdg=1,
    ),
)


@patch("src.app.services.sql_db.session_maker")
@patch("src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True))
@patch(search_pipeline_path, new=mock.MagicMock())
@patch(
    f"{search_pipeline_path}.get_collections",
    new=mock.AsyncMock(return_value=("collection_fr_model", "collection_en_model")),
)
@patch(
    f"{search_pipeline_path}._get_info_from_collection_alias",
    new=mock.MagicMock(return_value=mocked_collection),
)
class SearchTests(IsolatedAsyncioTestCase):
    @patch(
        "src.app.api.api_v1.endpoints.search.search_items_base",
        new=mock.AsyncMock(return_value=[mocked_document]),
    )
    @patch(
        "src.app.services.search.SearchService.search_group_by_document",
        new=mock.AsyncMock(return_value=["document_1", "document_2"]),
    )
    def test_search_items_success(self, *mocks):
        """Test successful search_items response"""

        response = client.post(
            f"{settings.API_V1_STR}/search/collections/collection_fr_model?query=français&nb_results=10",
            headers={"X-API-Key": "test"},  # noqa: E501
        )

        # Assert: Check that the response is as expected
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [
                {
                    "score": 0.9,
                    "payload": {
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
                }
            ],
        )

    @patch(
        "src.app.api.api_v1.endpoints.search.search_items_base",
        new=mock.AsyncMock(return_value=[mocked_document]),
    )
    @patch(f"{search_pipeline_path}.search_group_by_document")
    def test_search_items_no_query(self, *mocks):
        """Test search_items when no query is provided"""

        # Act: Make a test request to the /collections/{collection_query} endpoint without query
        response = client.post(
            f"{settings.API_V1_STR}/search/collections/collection_fr_model",  # noqa: E501
            json={"nb_results": 10},
            headers={"X-API-Key": "test"},
        )

        # Assert: Check that the response is as expected
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": {"code": "EMPTY_QUERY", "message": "Empty query"}},
        )

    @patch(
        f"{search_pipeline_path}.get_collections",
        new=mock.AsyncMock(return_value=("NOTTOTO_fr_model")),
    )
    async def test_search_collection_not_found(self, *mocks):
        with self.assertRaises(CollectionNotFoundError):
            response = client.post(
                f"{settings.API_V1_STR}/search/collections/toto?query=français&nb_results=10",
                headers={"X-API-Key": "test"},
            )

            self.assertEqual(response.status_code, 404)
            self.assertEqual(
                response.json().get("detail")["code"],
                "COLL_NOT_FOUND",
            )

    @patch(
        f"{search_pipeline_path}.get_model",
        new=mock.MagicMock(
            side_effect=ModelNotFoundError("Model not found", "MODEL_NOT_FOUND")
        ),
    )
    async def test_search_model_not_found(self, *mocks):
        with self.assertRaises(ModelNotFoundError):
            response = client.post(
                f"{settings.API_V1_STR}/search/collections/collection_fr_model?query=français&nb_results=10",  # noqa: E501
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
        f"{search_pipeline_path}.get_collections_aliases_by_language",
        new=mock.AsyncMock(return_value=()),
    )
    async def test_search_all_slices_no_collections(self, *mocks):
        with self.assertRaises(CollectionNotFoundError):
            response = client.post(
                f"{settings.API_V1_STR}/search/by_slices?nb_results=10",
                json={
                    "query": "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne"
                },  # noqa: E501
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 404)
            self.assertEqual(
                response.json().get("detail")["code"],
                "COLL_NOT_FOUND",
            )

    @patch(
        f"{search_pipeline_path}.get_collections_aliases_by_language",
        return_value=("collection_fr_model"),
    )
    @patch(f"{search_pipeline_path}.get_collection_dict_with_embed")
    @patch(
        "src.app.services.search_helpers.parallel_search",
        return_value=mocked_scored_points,
    )
    async def test_search_all_slices_ok(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_slices?nb_results=10",
            json={
                "query": "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne"
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
        f"{search_pipeline_path}.get_collections_aliases_by_language",
        return_value=("collection_fr_model"),
    )
    @patch(f"{search_pipeline_path}.get_collection_dict_with_embed")
    @patch(
        "src.app.services.search_helpers.parallel_search",
        return_value=[],
    )
    async def test_search_all_slices_no_result(self, *mocks):
        with self.assertRaises(NoResultsError):
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
        f"{search_pipeline_path}.get_collections_aliases_by_language",
        new=mock.AsyncMock(return_value=()),
    )
    @patch(
        f"{search_pipeline_path}._get_info_from_collection_alias",
        new=mock.MagicMock(return_value=mocked_collection),
    )
    async def test_search_all_no_collections(self, *mocks):
        with self.assertRaises(CollectionNotFoundError):
            response = client.post(
                f"{settings.API_V1_STR}/search/by_document?nb_results=10",
                json={
                    "query": "une phrase plus longue pour tester la recherche en français. et voir ce que cela donne"
                },
                headers={"X-API-Key": "test"},
            )
            self.assertEqual(response.status_code, 404)
            self.assertEqual(
                response.json().get("detail")["code"],
                "COLL_NOT_FOUND",
            )

    @patch(
        f"{search_pipeline_path}.get_collections_aliases_by_language",
        return_value=["collection_fr_model"],
    )
    @patch(
        f"{search_pipeline_path}._get_info_from_collection_alias",
        new=mock.MagicMock(return_value=mocked_collection),
    )
    @patch(f"{search_pipeline_path}.embed_query")
    @patch(
        "src.app.services.search_helpers.parallel_search",
        return_value=[],
    )
    async def test_search_all_no_result(self, *mocks):
        with self.assertRaises(NoResultsError):
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
