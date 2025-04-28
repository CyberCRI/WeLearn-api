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
)
from src.app.services.search import sort_slices_using_mmr
from src.main import app

client = TestClient(app)

search_pipeline_path = "src.app.services.search.SearchService"
parallel_search_path = "src.app.api.api_v1.endpoints.search.parallel_search"

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
    new=mock.AsyncMock(
        return_value=("collection_welearn_fr_model", "collection_en_model")
    ),
)
@patch(
    f"{search_pipeline_path}._get_info_from_collection_name",
    new=mock.MagicMock(return_value=mocked_collection),
)
class SearchTests(IsolatedAsyncioTestCase):
    @patch(
        "src.app.services.search.SearchService.search_handler",
        new=mock.AsyncMock(return_value=mocked_scored_points),
    )
    async def test_search_items_success(self, *mocks):
        """Test successful search_items response"""
        long_query = "français with a very long sentence to test what you are saying and if the issue is the size of the string"  # noqa: E501

        response = client.post(
            f"{settings.API_V1_STR}/search/collections/collection_welearn_fr_model?query={long_query}&nb_results=10",
            headers={"X-API-Key": "test"},  # noqa: E501
        )

        # Assert: Check that the response is as expected
        self.assertEqual(response.status_code, 200)

    @patch(f"{search_pipeline_path}.search_group_by_document")
    def test_search_items_no_query(self, *mocks):
        """Test search_items when no query is provided"""

        # Act: Make a test request to the /collections/{collection_query} endpoint without query
        response = client.post(
            f"{settings.API_V1_STR}/search/collections/collection_welearn_fr_model",  # noqa: E501
            json={"nb_results": 10},
            headers={"X-API-Key": "test"},
        )

        # Assert: Check that the response is as expected
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

    # patch should raise
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

    @patch(
        f"{search_pipeline_path}.search_handler",
        return_value=[
            {
                "id": 1,
                "score": 0.7513504,
                "payload": {
                    "document_corpus": "conversation",
                    "document_desc": "More and more evidence has accumulated which shows that changes in global and regional climate over the last 50 years are almost entirely due to human influence.",
                    "document_details": {
                        "authors": [
                            {"misc": "", "name": "Mark New"},
                            {"misc": "", "name": "University of Cape Town"},
                        ],
                        "duration": "316",
                        "readability": "56.49",
                        "source": "africa",
                    },
                    "document_id": "af2ca7b2-1011-4b4d-828e-9678597fd255",
                    "document_lang": "en",
                    "document_sdg": [13],
                    "document_title": "Climate explained: how much of climate change is natural? How much is man-made?",
                    "document_url": "https://theconversation.com/climate-explained-how-much-of-climate-change-is-natural-how-much-is-man-made-123604",
                    "slice_content": "The Intergovernmental Panel on Climate Change defines climate change as: a change in the state of the climate that can be identified by changes in the mean and/or the variability of its properties and that persists for an extended period, typically decades or longer. The causes of climate change can be any combination of: Internal variability in the climate system, when various components of the climate system – like the atmosphere and ocean – vary on their own to cause fluctuations in climatic conditions, such as temperature or rainfall. These internally-driven changes generally happen over decades or longer; shorter variations such as those related to El Niño fall in the bracket of climate variability, not climate change. Natural external causes such as increases or decreases in volcanic activity or solar radiation. For example, every 11 years or so, the Sun’s magnetic field completely flips and this can cause small fluctuations in global temperature, up to about 0.2 degrees. On longer time scales – tens to hundreds of millions of years – geological processes can drive changes in the climate, due to shifting continents and mountain building. Human influence through greenhouse gases (gases that trap heat in the atmosphere such as carbon dioxide and methane), other particles released into the air (which absorb or reflect sunlight such as soot and aerosols) and land-use change (which affects how much sunlight is absorbed on land surfaces and also how much carbon dioxide and methane is absorbed and released by vegetation and soils). What changes have been detected?\n\nThe Intergovernmental Panel on Climate Change’s recent report showed that, on average, the global surface air temperature has risen by 1°C since the beginning of significant industrialisation (which roughly started in the 1850s). And it is increasing at ever faster rates, currently 0.2°C per decade, because the concentrations of greenhouse gases in the atmosphere have themselves been increasing ever faster. The oceans are warming as well. In fact, about 90% of the extra heat trapped in the atmosphere by greenhouse gases is being absorbed by the oceans. A warmer atmosphere and oceans are causing dramatic changes, including steep decreases in Arctic summer sea ice which is profoundly impacting arctic marine ecosystems, increasing sea level rise which is inundating low lying coastal areas such as Pacific island atolls, and an increasing frequency of many climate extremes such as drought and heavy rain, as well as disasters where climate is an important driver, such as wildfire, flooding and landslides. Multiple lines of evidence, using different methods, show that human influence is the only plausible explanation for the patterns and magnitude of changes that have been detected. This human influence is largely due to our activities that release greenhouse gases, such as carbon dioxide and methane, as well sunlight absorbing soot. The main sources of these warming gases and particles are fossil fuel burning, cement production, land cover change (especially deforestation) and agriculture. Weather attribution Most of us will struggle to pick up slow changes in the climate.",
                    "slice_sdg": 13,
                },
            }
        ],
    )
    async def test_search_all_slices_ok(self, *mocks):
        response = client.post(
            f"{settings.API_V1_STR}/search/by_slices",
            json={
                "query": "Comment est-ce que les gouvernements font pour suivre ces conseils et les mettre en place ?",
                "relevance_factor": 0.75,
                "sdg_filter": [],
                "corpora": [],
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
        self.assertEqual(response.status_code, 404)


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
        self.assertEqual(response.status_code, 404)

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
