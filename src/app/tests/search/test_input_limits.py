import uuid
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.app.core.config import settings
from src.app.data_collection.services.data_collection import get_data_collection_service
from src.app.search.domain.models import EnhancedSearchQuery
from src.app.search.services.search import get_search_service
from src.app.shared.domain.documents import MAX_DOCUMENTS_IDS
from src.main import app

SEARCH = f"{settings.API_V1_STR}/search"
HEADERS = {"X-API-Key": "test"}


@patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=MagicMock(return_value=True),
)
class SearchInputLimitsTests(TestCase):
    """Out-of-range inputs are rejected with 422 before any search runs."""

    def setUp(self):
        self.client = TestClient(app)
        search_service = MagicMock()
        search_service.search_handler.side_effect = AssertionError("must not run")
        app.dependency_overrides[get_search_service] = lambda: search_service
        self.addCleanup(app.dependency_overrides.pop, get_search_service)
        app.dependency_overrides[get_data_collection_service] = lambda: MagicMock()
        self.addCleanup(app.dependency_overrides.pop, get_data_collection_service)
        corpora = patch(
            "src.app.search.services.search.get_visible_corpora_sync",
            return_value=("conversation",),
        )
        corpora.start()
        self.addCleanup(corpora.stop)

    def post(self, path, **kwargs):
        return self.client.post(f"{SEARCH}{path}", headers=HEADERS, **kwargs)

    def assert_rejected(self, response, field):
        self.assertEqual(response.status_code, 422)
        self.assertIn(field, [err["loc"][-1] for err in response.json()["detail"]])

    def test_collection_nb_results_above_100(self):
        response = self.post("/collections/conversation?query=water&nb_results=101")
        self.assert_rejected(response, "nb_results")

    def test_collection_nb_results_below_1(self):
        response = self.post("/collections/conversation?query=water&nb_results=0")
        self.assert_rejected(response, "nb_results")

    def test_relevance_factor_out_of_range(self):
        for path in ("/by_slices", "/by_document", "/multiple_by_slices"):
            for value in (-0.1, 1.1):
                with self.subTest(path=path, value=value):
                    response = self.post(
                        f"{path}?relevance_factor={value}", json={"query": "water"}
                    )
                    self.assert_rejected(response, "relevance_factor")

    def test_influence_factor_out_of_range(self):
        for path in ("/by_slices", "/by_document"):
            for value in (-1, 10.5):
                with self.subTest(path=path, value=value):
                    response = self.post(
                        f"{path}?influence_factor={value}", json={"query": "water"}
                    )
                    self.assert_rejected(response, "influence_factor")

    def test_multiple_by_slices_too_many_queries(self):
        response = self.post("/multiple_by_slices", json={"query": ["q"] * 11})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "TOO_MANY_QUERIES")

    def test_by_ids_rejects_non_uuid(self):
        with patch(
            "src.app.search.api.router.get_documents_payload_by_ids_sync"
        ) as query:
            response = self.post("/documents/by_ids", json=["not-a-uuid"])

        self.assertEqual(response.status_code, 422)
        query.assert_not_called()

    def test_by_ids_rejects_too_many_ids(self):
        ids = [str(uuid.uuid4()) for _ in range(MAX_DOCUMENTS_IDS + 1)]
        with patch(
            "src.app.search.api.router.get_documents_payload_by_ids_sync"
        ) as query:
            response = self.post("/documents/by_ids", json=ids)

        self.assertEqual(response.status_code, 422)
        query.assert_not_called()

    def test_by_ids_passes_ids_as_strings(self):
        doc_id = uuid.uuid4()
        with patch(
            "src.app.search.api.router.get_documents_payload_by_ids_sync",
            return_value=[],
        ) as query:
            response = self.post("/documents/by_ids", json=[str(doc_id)])

        self.assertEqual(response.status_code, 200)
        query.assert_called_once_with([str(doc_id)])


@patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=MagicMock(return_value=True),
)
class BibliographyInputLimitsTests(TestCase):
    def test_too_many_documents_ids(self):
        ids = [str(uuid.uuid4()) for _ in range(MAX_DOCUMENTS_IDS + 1)]
        with patch("src.app.bibliography.api.router.get_documents_by_ids") as query:
            response = TestClient(app).post(
                f"{settings.API_V1_STR}/bibliography/export_bibliography",
                json={"documents_ids": ids},
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 422)
        query.assert_not_called()


class EnhancedSearchQueryBoundsTests(TestCase):
    def test_bounds(self):
        for field, value in (
            ("nb_results", 101),
            ("nb_results", 0),
            ("relevance_factor", 1.5),
            ("relevance_factor", -0.5),
            ("influence_factor", 11),
            ("influence_factor", -1),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    EnhancedSearchQuery(query="q", **{field: value})

    def test_limits_are_inclusive(self):
        qp = EnhancedSearchQuery(
            query="q", nb_results=100, relevance_factor=1, influence_factor=10
        )
        self.assertEqual(qp.nb_results, 100)
