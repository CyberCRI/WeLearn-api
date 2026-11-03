import json
import unittest
import uuid
from unittest.mock import MagicMock, call, patch

from fastapi import HTTPException
from qdrant_client.http.models import ScoredPoint
from welearn_database.data.models import QuestionType

from src.app.data_collection.services.data_collection import DataCollection, _cache
from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)
from src.app.tutor.domain.models import (
    ExtractorOutput,
    SyllabusFeedback,
    SyllabusResponseAgent,
    TutorSyllabusRequest,
)

MODULE = "src.app.data_collection.services.data_collection"


async def fake_run_in_threadpool(func, *args, **kwargs):
    return func(*args, **kwargs)


def _doc() -> EnhancedDocumentSlice:
    return EnhancedDocumentSlice(
        score=0.5,
        payload=EnhancedDocumentSlicePayloadModel(
            document_corpus="corpus",
            document_desc="desc",
            document_details={},
            document_id=uuid.uuid4(),
            document_lang="en",
            document_sdg=[1],
            document_title="title",
            document_url="https://example.org",
            slice_content="content",
            slice_sdg=1,
        ),
    )


def _syllabus_request(documents=None) -> TutorSyllabusRequest:
    return TutorSyllabusRequest(
        course_title="course",
        level="beginner",
        duration="10h",
        description="a course",
        documents=documents or [],
        extracts=[ExtractorOutput(summary="summary", themes=[])],
        nb_results=1,
    )


def _make_service(collect: bool) -> DataCollection:
    _cache["is_campaign_active"] = None
    _cache["expires"] = None
    with patch(f"{MODULE}.get_current_data_collection_campaign") as campaign:
        campaign.return_value = MagicMock(is_active=collect)
        return DataCollection(origin="workshop.example.com")


class _ServiceTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch(
            f"{MODULE}.run_in_threadpool", side_effect=fake_run_in_threadpool
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.get_user = self._patch("get_user_from_session_id")
        self.write_user_query = self._patch("write_user_query")
        self.write_chat_answer = self._patch("write_chat_answer")
        self.write_returned_docs = self._patch("write_returned_docs")
        self.write_filters_search = self._patch("write_filters_search")
        self.get_last_conv = self._patch("get_last_syllabus_conversation_id")
        self.get_last_syllabus_id = self._patch("get_last_syllabus_id_for_user")
        self.update_retrieved = self._patch("update_syllabus_retrieved_status")
        self.register_form = self._patch("register_form_answers_sync")

        self.user_id = uuid.uuid4()
        self.get_user.return_value = self.user_id
        self.dc = _make_service(collect=True)

    def tearDown(self):
        _cache["is_campaign_active"] = None
        _cache["expires"] = None

    def _patch(self, name):
        patcher = patch(f"{MODULE}.{name}")
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def _writers(self):
        return [
            self.write_user_query,
            self.write_chat_answer,
            self.write_returned_docs,
            self.write_filters_search,
            self.update_retrieved,
        ]

    def assert_nothing_written(self):
        for writer in self._writers():
            writer.assert_not_called()


class TestCampaignStateCache(unittest.TestCase):
    def tearDown(self):
        _cache["is_campaign_active"] = None
        _cache["expires"] = None

    @patch(f"{MODULE}.get_current_data_collection_campaign")
    def test_campaign_state_is_cached(self, mock_campaign):
        mock_campaign.return_value = MagicMock(is_active=True)
        _cache["is_campaign_active"] = None
        _cache["expires"] = None

        first = DataCollection(origin="workshop.example.com")
        second = DataCollection(origin="workshop.example.com")

        self.assertTrue(first.should_collect)
        self.assertTrue(second.should_collect)
        mock_campaign.assert_called_once()

    @patch(f"{MODULE}.get_current_data_collection_campaign", return_value=None)
    def test_no_campaign_is_cached_as_inactive(self, mock_campaign):
        _cache["is_campaign_active"] = None
        _cache["expires"] = None

        DataCollection(origin="workshop.example.com")
        second = DataCollection(origin="workshop.example.com")

        self.assertFalse(second.should_collect)
        mock_campaign.assert_called_once()

    @unittest.expectedFailure
    @patch(f"{MODULE}.get_current_data_collection_campaign")
    def test_inactive_campaign_is_cached_as_inactive(self, mock_campaign):
        # Known bug: the cached path returns `is_campaign_active is not None`,
        # so a cached False (campaign exists but is_active=False) reads as True.
        mock_campaign.return_value = MagicMock(is_active=False)
        _cache["is_campaign_active"] = None
        _cache["expires"] = None

        DataCollection(origin="workshop.example.com")
        second = DataCollection(origin="workshop.example.com")

        self.assertFalse(second.should_collect)


class TestDisabledCollection(_ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.dc = _make_service(collect=False)

    async def test_register_search_data_disabled(self):
        result = await self.dc.register_search_data(uuid.uuid4(), "q", [])
        self.assertIsNone(result)
        self.assert_nothing_written()

    async def test_register_syllabus_data_disabled(self):
        result = await self.dc.register_syllabus_data(
            uuid.uuid4(), _syllabus_request(), "answer", "syllabus_creation"
        )
        self.assertIsNone(result)
        self.assert_nothing_written()

    async def test_register_syllabus_update_disabled(self):
        result = await self.dc.register_syllabus_update(uuid.uuid4(), "content")
        self.assertIsNone(result)
        self.assert_nothing_written()

    async def test_register_chat_data_disabled(self):
        result = await self.dc.register_chat_data(uuid.uuid4(), "q", None, "answer", [])
        self.assertEqual(result, (None, None))
        self.assert_nothing_written()

    async def test_register_document_click_disabled(self):
        with patch(f"{MODULE}.update_returned_document_click") as update_click:
            await self.dc.register_document_click(uuid.uuid4(), uuid.uuid4())
        update_click.assert_not_called()

    async def test_register_syllabus_download_disabled(self):
        await self.dc.register_syllabus_download(uuid.uuid4())
        self.get_user.assert_not_called()
        self.assert_nothing_written()


class TestMissingSessionOrUser(_ServiceTestBase):
    def _calls(self, session_id):
        return {
            "search": lambda: self.dc.register_search_data(session_id, "q", []),
            "syllabus_data": lambda: self.dc.register_syllabus_data(
                session_id, _syllabus_request(), "answer", "syllabus_creation"
            ),
            "syllabus_update": lambda: self.dc.register_syllabus_update(
                session_id, "content"
            ),
            "syllabus_download": lambda: self.dc.register_syllabus_download(session_id),
        }

    async def test_missing_session_id_raises_401(self):
        for name, make_call in self._calls(None).items():
            with self.subTest(name):
                with self.assertRaises(HTTPException) as ctx:
                    await make_call()
                self.assertEqual(ctx.exception.status_code, 401)
                self.assertEqual(ctx.exception.detail["code"], "SESSION_ID_NOT_FOUND")
        self.get_user.assert_not_called()
        self.assert_nothing_written()

    async def test_unknown_user_raises_401(self):
        self.get_user.return_value = None
        session_id = uuid.uuid4()
        for name, make_call in self._calls(session_id).items():
            with self.subTest(name):
                with self.assertRaises(HTTPException) as ctx:
                    await make_call()
                self.assertEqual(ctx.exception.status_code, 401)
                self.assertEqual(ctx.exception.detail["code"], "USER_NOT_FOUND")
        self.get_user.assert_called_with(session_id)
        self.assert_nothing_written()


class TestRegisterSearchData(_ServiceTestBase):
    async def test_writes_query_docs_and_filters(self):
        conversation_id, msg_id = uuid.uuid4(), uuid.uuid4()
        self.write_user_query.return_value = (conversation_id, msg_id)
        session_id = uuid.uuid4()
        results = [_doc(), ScoredPoint(id=1, version=0, score=0.1, payload={})]

        result = await self.dc.register_search_data(
            session_id, "query", results, sdg_filter=[1, 2], corpora=["c"]
        )

        self.assertEqual(result, msg_id)
        self.get_user.assert_called_once_with(session_id)
        self.write_user_query.assert_called_once_with(
            self.user_id, "query", None, "search"
        )
        self.write_returned_docs.assert_called_once_with(msg_id, results)
        self.write_filters_search.assert_called_once_with(msg_id, [1, 2], ["c"])

    async def test_no_filters_skips_filter_write(self):
        msg_id = uuid.uuid4()
        self.write_user_query.return_value = (uuid.uuid4(), msg_id)

        result = await self.dc.register_search_data(
            uuid.uuid4(), "query", [], feature="tutor_search"
        )

        self.assertEqual(result, msg_id)
        self.write_user_query.assert_called_once_with(
            self.user_id, "query", None, "tutor_search"
        )
        self.write_returned_docs.assert_called_once_with(msg_id, [])
        self.write_filters_search.assert_not_called()


class TestRegisterSyllabusData(_ServiceTestBase):
    async def test_creation_serialises_request_and_writes_docs(self):
        conversation_id = uuid.uuid4()
        answer_id = uuid.uuid4()
        self.write_user_query.return_value = (conversation_id, uuid.uuid4())
        self.write_chat_answer.return_value = answer_id
        docs = [_doc(), _doc()]

        result = await self.dc.register_syllabus_data(
            uuid.uuid4(), _syllabus_request(docs), "answer", "syllabus_creation"
        )

        self.assertEqual(result, answer_id)
        self.get_last_conv.assert_not_called()
        user_id, user_data, conv, feature = self.write_user_query.call_args.args
        self.assertEqual(
            (user_id, conv, feature), (self.user_id, None, "syllabus_creation")
        )
        self.assertEqual(
            json.loads(user_data),
            {
                "course_title": "course",
                "level": "beginner",
                "duration": "10h",
                "description": "a course",
            },
        )
        self.write_chat_answer.assert_called_once_with(
            self.user_id, "answer", None, conversation_id, "syllabus_creation"
        )
        self.write_returned_docs.assert_called_once_with(answer_id, docs, True)

    async def test_feedback_reuses_last_conversation(self):
        last_conv = uuid.uuid4()
        answer_id = uuid.uuid4()
        self.get_last_conv.return_value = last_conv
        self.write_user_query.return_value = (last_conv, uuid.uuid4())
        self.write_chat_answer.return_value = answer_id
        feedback = SyllabusFeedback(
            syllabus=[SyllabusResponseAgent(content="syllabus")],
            documents=[],
            extracts=[],
            feedback="make it shorter",
        )

        result = await self.dc.register_syllabus_data(
            uuid.uuid4(), feedback, "new answer", "syllabus_feedback"
        )

        self.assertEqual(result, answer_id)
        self.get_last_conv.assert_called_once_with(self.user_id)
        self.write_user_query.assert_called_once_with(
            self.user_id, "make it shorter", last_conv, "syllabus_feedback"
        )
        self.write_chat_answer.assert_called_once_with(
            self.user_id, "new answer", None, last_conv, "syllabus_feedback"
        )
        self.write_returned_docs.assert_not_called()


class TestRegisterSyllabusUpdate(_ServiceTestBase):
    async def test_no_previous_conversation_writes_nothing(self):
        self.get_last_conv.return_value = None

        result = await self.dc.register_syllabus_update(uuid.uuid4(), "content")

        self.assertIsNone(result)
        self.get_last_conv.assert_called_once_with(self.user_id)
        self.write_user_query.assert_not_called()


class TestRegisterSyllabusDownload(_ServiceTestBase):
    async def test_no_syllabus_does_not_update(self):
        self.get_last_syllabus_id.return_value = None

        await self.dc.register_syllabus_download(uuid.uuid4())

        self.get_last_syllabus_id.assert_called_once_with(self.user_id)
        self.update_retrieved.assert_not_called()


class TestRegisterFormAnswer(_ServiceTestBase):
    async def test_forwards_answer_even_when_collection_disabled(self):
        self.dc = _make_service(collect=False)
        session_id = uuid.uuid4()

        await self.dc.register_form_answer(
            "form", "question?", "answer", QuestionType.CHECKBOX, session_id
        )

        self.assertEqual(
            self.register_form.call_args_list,
            [call("form", "question?", "answer", QuestionType.CHECKBOX, session_id)],
        )
