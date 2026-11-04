import time
import uuid
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.app.baml_client.async_client import types
from src.app.core.config import settings
from src.app.data_collection.services.data_collection import get_data_collection_service
from src.app.search.services.search import get_search_service
from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)
from src.app.shared.domain.exceptions import NoResultsError
from src.app.tutor.domain.models import SyllabusResponseAgent, UserInput
from src.main import app

ROUTER = "src.app.tutor.api.router"
BASE = f"{settings.API_V1_STR}/tutor"
HEADERS = {"X-API-Key": "test"}
SESSION_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")

COURSE_METADATA = {
    "discipline": "Biology",
    "topic": "Ecology",
    "level": "L2",
    "num_sessions": 4,
    "session_duration": 2.0,
    "session_type": "TD",
    "session_mode": "PRESENTIEL",
    "class_size": 30,
    "output_language": "en",
}

OBJECTIVES = {
    "objectives": [
        {"number": 1, "text": "Understand ecosystems", "bloom_level": "understand"},
        {"number": 2, "text": "Analyse food webs", "bloom_level": None},
    ]
}


def make_document() -> EnhancedDocumentSlice:
    return EnhancedDocumentSlice(
        score=0.9,
        id=uuid.uuid4(),
        payload=EnhancedDocumentSlicePayloadModel(
            document_corpus="corpus",
            document_desc="desc",
            document_details={},
            document_id=uuid.uuid4(),
            document_lang="en",
            document_sdg=[13],
            document_title="Climate",
            document_url="https://example.org",
            slice_content="content",
            slice_sdg=13,
        ),
    )


class TutorApiTestCase(TestCase):
    """Base class: mocks the API key check and the injected dependencies."""

    def setUp(self):
        patcher = patch(
            "src.app.shared.infra.security.check_api_key_sync",
            new=MagicMock(return_value=True),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.data_collection = MagicMock()
        self.data_collection.register_syllabus_data = AsyncMock(
            return_value=uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        )
        self.data_collection.register_syllabus_update = AsyncMock()
        self.search_service = MagicMock()

        app.dependency_overrides[get_data_collection_service] = (
            lambda: self.data_collection
        )
        app.dependency_overrides[get_search_service] = lambda: self.search_service
        self.addCleanup(app.dependency_overrides.pop, get_data_collection_service)
        self.addCleanup(app.dependency_overrides.pop, get_search_service)

        self.client = TestClient(app)
        self.safe_client = TestClient(app, raise_server_exceptions=False)


class TestMasFilesContent(TutorApiTestCase):
    URL = f"{BASE}/mas_files/content"

    def post_files(self, client=None, **params):
        return (client or self.client).post(
            self.URL,
            params=params,
            files=[
                ("files", ("a.txt", b"first", "text/plain")),
                ("files", ("b.txt", b"second", "text/plain")),
            ],
            headers=HEADERS,
        )

    @patch(f"{ROUTER}.b.FileSummaryExtractor", new_callable=AsyncMock)
    @patch(f"{ROUTER}.get_files_content", new_callable=AsyncMock)
    def test_extracts_summary(self, mock_files, mock_extractor):
        mock_files.return_value = ["first", "second"]
        mock_extractor.return_value = types.FileSummary(
            detected_document_type="syllabus",
            confidence=0.8,
            summary="A summary",
            objectives=["obj"],
        )

        response = self.post_files(lang="fr", mode="mode_2")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["detected_document_type"], "syllabus")
        self.assertEqual(body["objectives"], ["obj"])
        self.assertEqual(len(mock_files.await_args.args[0]), 2)
        mock_extractor.assert_awaited_once_with(
            mode="syllabus",
            file_content="first__DOCUMENT_SEPARATOR__second",
            output_language="fr",
        )

    @patch(f"{ROUTER}.b.FileSummaryExtractor", new_callable=AsyncMock)
    @patch(f"{ROUTER}.get_files_content", new_callable=AsyncMock)
    def test_default_mode_is_article(self, mock_files, mock_extractor):
        mock_files.return_value = ["only"]
        mock_extractor.return_value = types.FileSummary(
            detected_document_type="article", confidence=1.0, objectives=[]
        )

        response = self.post_files()

        self.assertEqual(response.status_code, 200)
        mock_extractor.assert_awaited_once_with(
            mode="article", file_content="only", output_language="en"
        )

    @patch(f"{ROUTER}.b.FileSummaryExtractor", new_callable=AsyncMock)
    @patch(f"{ROUTER}.get_files_content", new_callable=AsyncMock)
    def test_mode_3_has_no_mode(self, mock_files, mock_extractor):
        mock_files.return_value = ["x"]
        mock_extractor.return_value = types.FileSummary(
            detected_document_type="other", confidence=0.1, objectives=[]
        )

        self.post_files(mode="mode_3")

        self.assertIsNone(mock_extractor.await_args.kwargs["mode"])

    @patch(f"{ROUTER}.b.FileSummaryExtractor", new_callable=AsyncMock)
    @patch(f"{ROUTER}.get_files_content", new_callable=AsyncMock)
    def test_extractor_error_is_reraised(self, mock_files, mock_extractor):
        mock_files.return_value = ["x"]
        mock_extractor.side_effect = RuntimeError("llm down")

        response = self.post_files(client=self.safe_client)

        self.assertEqual(response.status_code, 500)
        mock_extractor.assert_awaited_once()

    @patch(f"{ROUTER}.get_files_content", new_callable=AsyncMock)
    def test_rejects_invalid_api_key(self, mock_files):
        with patch(
            "src.app.shared.infra.security.check_api_key_sync",
            new=MagicMock(return_value=False),
        ):
            response = self.client.post(
                self.URL,
                files=[("files", ("a.txt", b"x", "text/plain"))],
                headers=HEADERS,
            )

        self.assertEqual(response.status_code, 401)
        mock_files.assert_not_awaited()


class TestSearchExtracts(TutorApiTestCase):
    URL = f"{BASE}/search_extracts"

    @patch(f"{ROUTER}.search_multi_inputs", new_callable=AsyncMock)
    def test_returns_documents(self, mock_search):
        docs = [make_document(), make_document()]
        mock_search.return_value = docs

        response = self.client.post(
            self.URL,
            params={"nb_results": 5},
            json={"summaries": ["climate", "biodiversity"]},
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["nb_results"], 2)
        self.assertEqual(body["extracts"], [])
        self.assertEqual(
            [d["payload"]["document_title"] for d in body["documents"]],
            ["Climate", "Climate"],
        )
        self.assertNotIn("vector", body["documents"][0])

        kwargs = mock_search.await_args.kwargs
        self.assertEqual(kwargs["qp"].query, ["climate", "biodiversity"])
        self.assertEqual(kwargs["qp"].nb_results, 5)
        self.assertIsNone(kwargs["qp"].sdg_filter)
        self.assertIsNone(kwargs["qp"].corpora)
        self.assertIs(kwargs["callback_function"], self.search_service.search_handler)

    @patch(f"{ROUTER}.search_multi_inputs", new_callable=AsyncMock)
    def test_empty_results(self, mock_search):
        mock_search.return_value = []

        response = self.client.post(
            self.URL, json={"summaries": ["nothing"]}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"extracts": [], "nb_results": 0, "documents": []}
        )
        self.assertEqual(mock_search.await_args.kwargs["qp"].nb_results, 15)

    @patch(f"{ROUTER}.search_multi_inputs", new_callable=AsyncMock)
    def test_none_results(self, mock_search):
        mock_search.return_value = None

        response = self.client.post(
            self.URL, json={"summaries": ["nothing"]}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["nb_results"], 0)

    @patch(f"{ROUTER}.search_multi_inputs", new_callable=AsyncMock)
    def test_no_results_error_returns_404(self, mock_search):
        mock_search.side_effect = NoResultsError()

        response = self.client.post(
            self.URL, json={"summaries": ["nothing"]}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(), {"extracts": [], "nb_results": 0, "documents": []}
        )
        mock_search.assert_awaited_once()

    @patch(f"{ROUTER}.search_multi_inputs", new_callable=AsyncMock)
    def test_client_error_is_not_retried(self, mock_search):
        mock_search.side_effect = HTTPException(status_code=400, detail="bad")

        start = time.monotonic()
        response = self.client.post(
            self.URL, json={"summaries": ["climate"]}, headers=HEADERS
        )
        elapsed = time.monotonic() - start

        self.assertEqual(response.status_code, 400)
        mock_search.assert_awaited_once()
        self.assertLess(elapsed, 1.0)

    @patch(f"{ROUTER}.search_multi_inputs", new_callable=AsyncMock)
    def test_unexpected_error_is_not_retried(self, mock_search):
        mock_search.side_effect = ValueError("boom")

        start = time.monotonic()
        response = self.safe_client.post(
            self.URL, json={"summaries": ["climate"]}, headers=HEADERS
        )
        elapsed = time.monotonic() - start

        self.assertEqual(response.status_code, 500)
        mock_search.assert_awaited_once()
        self.assertLess(elapsed, 1.0)

    def test_invalid_body(self):
        response = self.client.post(
            self.URL, json={"summaries": "not a list"}, headers=HEADERS
        )

        self.assertEqual(response.status_code, 422)


class TestCreateSyllabus(TutorApiTestCase):
    URL = f"{BASE}/syllabus"

    def body(self):
        return {
            "extracts": [{"summary": "s", "themes": [{"theme": "t"}]}],
            "nb_results": 1,
            "documents": [make_document().model_dump(mode="json")],
            "course_title": "Ecology 101",
        }

    @patch(f"{ROUTER}.tutor_manager", new_callable=AsyncMock)
    def test_returns_syllabus_and_registers_data(self, mock_manager):
        mock_manager.return_value = [
            SyllabusResponseAgent(content="draft", source="a"),
            SyllabusResponseAgent(content="final", source="b"),
        ]

        response = self.client.post(
            self.URL,
            params={"lang": "fr"},
            json=self.body(),
            headers={**HEADERS, "X-Session-Id": str(SESSION_ID)},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual([s["content"] for s in body["syllabus"]], ["draft", "final"])
        self.assertEqual(
            body["syllabus_message_id"], "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )
        self.assertEqual(len(body["documents"]), 1)
        self.assertEqual(body["extracts"][0]["summary"], "s")

        args = mock_manager.await_args
        self.assertEqual(args.args[0].course_title, "Ecology 101")
        self.assertEqual(args.args[1], "fr")
        self.assertEqual(
            args.kwargs["trace_context"],
            {
                "endpoint": self.URL,
                "feature": "syllabus_creation",
                "session_id": str(SESSION_ID),
                "query_extracts_count": 1,
                "documents_count": 1,
            },
        )

        dc_kwargs = self.data_collection.register_syllabus_data.await_args.kwargs
        self.assertEqual(dc_kwargs["session_id"], SESSION_ID)
        self.assertEqual(dc_kwargs["agent_answer"], "final")
        self.assertEqual(dc_kwargs["feature"], "syllabus_creation")

    @patch(f"{ROUTER}.tutor_manager", new_callable=AsyncMock)
    def test_without_session(self, mock_manager):
        mock_manager.return_value = [SyllabusResponseAgent(content="only")]
        self.data_collection.register_syllabus_data.return_value = None

        response = self.client.post(self.URL, json=self.body(), headers=HEADERS)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["syllabus_message_id"])
        self.assertIsNone(mock_manager.await_args.kwargs["trace_context"]["session_id"])
        self.assertEqual(mock_manager.await_args.args[1], "en")
        self.assertIsNone(
            self.data_collection.register_syllabus_data.await_args.kwargs["session_id"]
        )

    @patch(f"{ROUTER}.tutor_manager", new_callable=AsyncMock)
    def test_generation_failure_returns_502(self, mock_manager):
        mock_manager.side_effect = RuntimeError("mistral down")

        response = self.client.post(self.URL, json=self.body(), headers=HEADERS)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            {
                "message": "Syllabus generation failed, please retry",
                "code": "SYLLABUS_GENERATION_FAILED",
            },
        )
        self.data_collection.register_syllabus_data.assert_not_awaited()


class TestSyllabusFeedback(TutorApiTestCase):
    URL = f"{BASE}/syllabus/feedback"

    def body(self):
        return {
            "syllabus": [{"content": "old syllabus"}],
            "documents": [],
            "extracts": [],
            "feedback": "more sessions please",
        }

    @patch(f"{ROUTER}.apply_feedback", new_callable=AsyncMock)
    def test_applies_feedback(self, mock_feedback):
        mock_feedback.return_value = "new syllabus"

        response = self.client.post(
            self.URL,
            json=self.body(),
            headers={**HEADERS, "X-Session-Id": str(SESSION_ID)},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["syllabus"], [{"content": "new syllabus", "source": "default"}]
        )
        self.assertIsNone(body["syllabus_message_id"])

        args = mock_feedback.await_args
        self.assertEqual(args.args[0].feedback, "more sessions please")
        self.assertEqual(
            args.kwargs["trace_context"],
            {
                "endpoint": self.URL,
                "feature": "syllabus_feedback",
                "session_id": str(SESSION_ID),
            },
        )
        self.data_collection.register_syllabus_data.assert_awaited_once()
        dc_kwargs = self.data_collection.register_syllabus_data.await_args.kwargs
        self.assertEqual(dc_kwargs["agent_answer"], "new syllabus")
        self.assertEqual(dc_kwargs["feature"], "syllabus_feedback")
        self.assertEqual(dc_kwargs["session_id"], SESSION_ID)

    @patch(f"{ROUTER}.apply_feedback", new_callable=AsyncMock)
    def test_feedback_error_is_reraised(self, mock_feedback):
        mock_feedback.side_effect = RuntimeError("boom")

        response = self.safe_client.post(self.URL, json=self.body(), headers=HEADERS)

        self.assertEqual(response.status_code, 500)
        self.data_collection.register_syllabus_data.assert_not_awaited()

    def test_missing_feedback_field(self):
        body = self.body()
        del body["feedback"]

        response = self.client.post(self.URL, json=body, headers=HEADERS)

        self.assertEqual(response.status_code, 422)


class TestSyllabusUserUpdate(TutorApiTestCase):
    URL = f"{BASE}/syllabus/user_update"

    def test_registers_update(self):
        response = self.client.post(
            self.URL,
            json={"syllabus": "# edited"},
            headers={**HEADERS, "X-Session-Id": str(SESSION_ID)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "user update registered"})
        self.data_collection.register_syllabus_update.assert_awaited_once_with(
            session_id=SESSION_ID, syllabus_content="# edited"
        )


class TestBamlStepEndpoints(TutorApiTestCase):
    def expected_metadata(self, **extra):
        return types.CourseMetadata(**COURSE_METADATA, **extra)

    @patch(f"{ROUTER}.b.GenerateCourseDescription", new_callable=AsyncMock)
    def test_description(self, mock_generate):
        mock_generate.return_value = types.CourseDescription(
            text="A course", word_count=2
        )
        metadata = {**COURSE_METADATA, "user_description": "my notes"}

        response = self.client.post(
            f"{BASE}/syllabus/description",
            json={
                "mode": "mode_1",
                "course_metadata": metadata,
                "context_text": "context",
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "A course", "word_count": 2})
        mock_generate.assert_awaited_once_with(
            mode="mode_1",
            metadata=self.expected_metadata(user_description="my notes"),
            context_text="context",
            output_language="en",
        )

    @patch(f"{ROUTER}.b.GenerateLearningObjectives", new_callable=AsyncMock)
    def test_learning_objectives(self, mock_generate):
        mock_generate.return_value = types.LearningObjectives(
            objectives=[types.LearningObjective(number=1, text="Obj")]
        )

        response = self.client.post(
            f"{BASE}/syllabus/learning_objectives",
            json={
                "description": "desc",
                "course_metadata": {**COURSE_METADATA, "user_description": "x"},
                "mode": "mode_2",
                "context_text": "ctx",
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"objectives": [{"number": 1, "text": "Obj", "bloom_level": None}]},
        )
        # user_description is not forwarded to this step
        mock_generate.assert_awaited_once_with(
            description="desc",
            mode="mode_2",
            metadata=self.expected_metadata(),
            context_text="ctx",
            output_language="en",
        )

    @patch(f"{ROUTER}.b.IntegrateSustainability", new_callable=AsyncMock)
    def test_sustainability_integration(self, mock_integrate):
        mock_integrate.return_value = types.SustainabilityIntegration(
            connections=[],
            integration_strategy="Weave SDG 13 into each session",
            resources_used=[],
        )

        response = self.client.post(
            f"{BASE}/syllabus/sustainability_integration",
            json={
                "description": "desc",
                "objectives": OBJECTIVES,
                "course_metadata": COURSE_METADATA,
                "sdg_resources": [],
                "mode": "mode_1",
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["integration_strategy"], "Weave SDG 13 into each session"
        )
        kwargs = mock_integrate.await_args.kwargs
        self.assertEqual(kwargs["description"], "desc")
        self.assertEqual(kwargs["metadata"], self.expected_metadata())
        self.assertEqual(kwargs["objectives"], types.LearningObjectives(**OBJECTIVES))
        self.assertEqual(kwargs["sdg_resources"], [])
        self.assertEqual(kwargs["mode"], "mode_1")
        self.assertEqual(kwargs["output_language"], "en")

    @patch(f"{ROUTER}.b.IntegrateSustainability", new_callable=AsyncMock)
    def test_sustainability_integration_with_resources(self, mock_integrate):
        resources = [
            {
                "text": "Wetlands store carbon.",
                "corpus": "ipcc",
                "description": "Report on wetlands",
                "sdg": [13, 15],
                "url": "https://example.org/wetlands",
                "title": "Wetlands and climate",
                "relevance_score": 0.87,
            },
            {
                "text": "Clean water access.",
                "corpus": "unesco",
                "description": "Water report",
                "sdg": [6],
                "url": "https://example.org/water",
                "title": "Water for all",
                "relevance_score": None,
            },
        ]
        expected_resources = [types.Document(**res) for res in resources]
        mock_integrate.return_value = types.SustainabilityIntegration(
            connections=[],
            integration_strategy="strategy",
            resources_used=expected_resources[:1],
        )

        response = self.client.post(
            f"{BASE}/syllabus/sustainability_integration",
            json={
                "description": "desc",
                "objectives": OBJECTIVES,
                "course_metadata": COURSE_METADATA,
                "sdg_resources": resources,
                "mode": "mode_1",
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["resources_used"], resources[:1])
        mock_integrate.assert_awaited_once()
        self.assertEqual(
            mock_integrate.await_args.kwargs["sdg_resources"], expected_resources
        )

    @patch(f"{ROUTER}.b.GenerateLearningOutcomes", new_callable=AsyncMock)
    def test_learning_outcomes(self, mock_generate):
        mock_generate.return_value = types.LearningOutcomes(
            outcomes=[
                types.LearningOutcome(
                    number=1,
                    text="Students map a food web",
                    related_objectives=[2],
                    assessment_method="Project",
                )
            ]
        )
        sustainability_map = {
            "connections": [],
            "integration_strategy": "strategy",
            "resources_used": [],
        }

        response = self.client.post(
            f"{BASE}/syllabus/learning_outcomes",
            json={
                "description": "desc",
                "objectives": OBJECTIVES,
                "course_metadata": COURSE_METADATA,
                "sustainability_map": sustainability_map,
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outcomes"][0]["related_objectives"], [2])
        mock_generate.assert_awaited_once_with(
            metadata=self.expected_metadata(),
            output_language="en",
            sustainability_map=types.SustainabilityIntegration(**sustainability_map),
            objectives=types.LearningObjectives(**OBJECTIVES),
        )

    @patch(f"{ROUTER}.b.MapCompetencies", new_callable=AsyncMock)
    def test_competency_map(self, mock_map):
        mock_map.return_value = types.CompetencyMappings(
            mappings=[
                types.CompetencyMapping(
                    outcome_number=1,
                    greencomp_competencies=["2.1"],
                    rationale="systems",
                )
            ]
        )

        response = self.client.post(
            f"{BASE}/syllabus/competency_map",
            json={
                "outcomes": [
                    {
                        "number": 1,
                        "text": "Map a food web",
                        "related_objectives": [1, 2],
                        "assessment_method": "Project",
                    }
                ],
                "output_language": "fr",
                "framework": {"2.1": "Systems thinking"},
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["mappings"][0]["greencomp_competencies"], ["2.1"]
        )
        # related objectives and assessment are dropped before calling BAML
        mock_map.assert_awaited_once_with(
            outcomes=types.LearningOutcomes(
                outcomes=[
                    types.LearningOutcome(
                        number=1,
                        text="Map a food web",
                        related_objectives=[],
                        assessment_method="",
                    )
                ]
            ),
            output_language="fr",
            greencomp_framework={"2.1": "Systems thinking"},
        )

    @patch(f"{ROUTER}.b.MapCompetencies", new_callable=AsyncMock)
    def test_competency_map_without_framework(self, mock_map):
        mock_map.return_value = types.CompetencyMappings(mappings=[])

        response = self.client.post(
            f"{BASE}/syllabus/competency_map",
            json={"outcomes": [], "output_language": "en"},
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"mappings": []})
        self.assertIsNone(mock_map.await_args.kwargs["greencomp_framework"])

    def test_description_invalid_session_mode(self):
        response = self.client.post(
            f"{BASE}/syllabus/description",
            json={
                "mode": "mode_1",
                "course_metadata": {**COURSE_METADATA, "session_mode": "ONLINE"},
                "context_text": "",
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 422)


class TestGenerateSyllabus(TutorApiTestCase):
    URL = f"{BASE}/api/generate"

    @patch(f"{ROUTER}.SyllabusOrchestrator")
    def test_success(self, mock_orchestrator_cls):
        description = types.CourseDescription(text="Course", word_count=1)
        objectives = types.LearningObjectives(
            objectives=[types.LearningObjective(number=1, text="Obj")]
        )
        orchestrator = mock_orchestrator_cls.return_value
        orchestrator.run = AsyncMock(
            return_value={"description": description, "objectives": objectives}
        )

        response = self.client.post(
            self.URL,
            json={
                "mode": "mode_1",
                "course_metadata": {**COURSE_METADATA, "user_description": "notes"},
                "context": "some context",
            },
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertIsNone(body["message"])
        self.assertEqual(
            body["validation"],
            {
                "passed": True,
                "major_issues": [],
                "minor_issues": [],
                "suggestions": [],
            },
        )
        self.assertEqual(body["description"], {"text": "Course", "word_count": 1})
        self.assertEqual(body["objectives"]["objectives"][0]["text"], "Obj")
        self.assertIsNone(body["sessions"])

        user_input = orchestrator.run.await_args.args[0]
        self.assertIsInstance(user_input, UserInput)
        self.assertEqual(user_input.mode, "mode_1")
        self.assertEqual(user_input.provided_description, "notes")
        self.assertIsNone(user_input.metadata.user_description)
        self.assertEqual(user_input.metadata.topic, "Ecology")
        self.assertEqual(
            orchestrator.run.await_args.kwargs, {"context_text": "some context"}
        )

    @patch(f"{ROUTER}.SyllabusOrchestrator")
    def test_without_user_description(self, mock_orchestrator_cls):
        orchestrator = mock_orchestrator_cls.return_value
        orchestrator.run = AsyncMock(return_value={})

        response = self.client.post(
            self.URL,
            json={"mode": "mode_1", "course_metadata": COURSE_METADATA},
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        user_input = orchestrator.run.await_args.args[0]
        self.assertIsNone(user_input.provided_description)
        self.assertIsNone(orchestrator.run.await_args.kwargs["context_text"])

    @patch(f"{ROUTER}.SyllabusOrchestrator")
    def test_orchestrator_error_returns_error_payload(self, mock_orchestrator_cls):
        orchestrator = mock_orchestrator_cls.return_value
        orchestrator.run = AsyncMock(side_effect=Exception("Phase 1 failed"))

        response = self.client.post(
            self.URL,
            json={"mode": "mode_1", "course_metadata": COURSE_METADATA},
            headers=HEADERS,
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["message"], "Phase 1 failed")
        self.assertEqual(
            body["validation"],
            {
                "passed": False,
                "major_issues": ["Phase 1 failed"],
                "minor_issues": [],
                "suggestions": [],
            },
        )
        self.assertIsNone(body["description"])
