import unittest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

from qdrant_client.http.models import ScoredPoint
from welearn_database.data.models import (
    AnalyticForm,
    ChatMessage,
    DataCollectionCampaignManagement,
    FilterUsedInQuery,
    InferredUser,
    QuestionType,
    ReturnedDocument,
    Session,
)

from src.app.data_collection.infra import queries
from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)
from src.app.tests.data_collection.sqlite_db import make_sqlite_session_maker


def _payload(document_id: uuid.UUID) -> dict:
    return {
        "document_corpus": "corpus",
        "document_desc": "desc",
        "document_details": {},
        "document_id": str(document_id),
        "document_lang": "en",
        "document_sdg": [1],
        "document_title": "title",
        "document_url": "https://example.org",
        "slice_content": "content",
        "slice_sdg": 1,
    }


def _enhanced_slice(document_id: uuid.UUID) -> EnhancedDocumentSlice:
    return EnhancedDocumentSlice(
        score=0.9,
        payload=EnhancedDocumentSlicePayloadModel(**_payload(document_id)),
    )


class DataCollectionQueriesTestBase(unittest.TestCase):
    def setUp(self):
        self.engine, self.s_maker = make_sqlite_session_maker()
        patcher = patch(
            "src.app.data_collection.infra.queries.session_maker", self.s_maker
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.engine.dispose)

        self.user_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=self.user_id))
            s.commit()

    def _add_message(self, feature: str | None, created_at: datetime, **kwargs):
        msg = ChatMessage(
            id=kwargs.pop("id", uuid.uuid4()),
            inferred_user_id=kwargs.pop("user_id", self.user_id),
            role="user",
            textual_content="content",
            conversation_id=kwargs.pop("conversation_id", uuid.uuid4()),
            original_feature_name=feature,
            created_at=created_at,
            **kwargs,
        )
        with self.s_maker() as s:
            s.add(msg)
            s.commit()
            return msg.id, msg.conversation_id

    def _all(self, model):
        with self.s_maker() as s:
            return s.query(model).all()


class TestRegisterFormAnswers(DataCollectionQueriesTestBase):
    def test_register_form_answers_writes_row(self):
        session_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(
                Session(
                    id=session_id,
                    inferred_user_id=self.user_id,
                    end_at=datetime.now() + timedelta(hours=1),
                )
            )
            s.commit()

        queries.register_form_answers_sync(
            form_name="satisfaction",
            question="Did you like it?",
            answer="yes",
            answer_type=QuestionType.TEXT,
            session_id=session_id,
        )

        rows = self._all(AnalyticForm)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].form_name, "satisfaction")
        self.assertEqual(rows[0].question, "Did you like it?")
        self.assertEqual(rows[0].answer, "yes")
        self.assertEqual(rows[0].answer_type, QuestionType.TEXT)
        self.assertEqual(rows[0].session_id, session_id)


class TestWriteUserQuery(DataCollectionQueriesTestBase):
    def test_generates_conversation_id_when_missing(self):
        conversation_id, msg_id = queries.write_user_query(
            self.user_id, "what is sdg 1?", None, "search"
        )

        self.assertIsInstance(conversation_id, uuid.UUID)
        rows = self._all(ChatMessage)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, msg_id)
        self.assertEqual(rows[0].conversation_id, conversation_id)
        self.assertEqual(rows[0].role, "user")
        self.assertEqual(rows[0].textual_content, "what is sdg 1?")
        self.assertEqual(rows[0].original_feature_name, "search")
        self.assertEqual(rows[0].inferred_user_id, self.user_id)

    def test_keeps_given_conversation_id(self):
        given = uuid.uuid4()

        conversation_id, _ = queries.write_user_query(self.user_id, "q", given)

        self.assertEqual(conversation_id, given)
        rows = self._all(ChatMessage)
        self.assertEqual(rows[0].conversation_id, given)
        self.assertIsNone(rows[0].original_feature_name)


class TestWriteChatAnswer(DataCollectionQueriesTestBase):
    def test_writes_answer_without_docs(self):
        conversation_id = uuid.uuid4()

        msg_id = queries.write_chat_answer(
            self.user_id, "the answer", None, conversation_id, "chat"
        )

        rows = self._all(ChatMessage)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, msg_id)
        self.assertEqual(rows[0].role, "assistant")
        self.assertEqual(rows[0].textual_content, "the answer")
        self.assertEqual(rows[0].conversation_id, conversation_id)
        self.assertEqual(rows[0].original_feature_name, "chat")
        self.assertEqual(self._all(ReturnedDocument), [])

    def test_writes_answer_with_docs(self):
        doc_a, doc_b = uuid.uuid4(), uuid.uuid4()

        msg_id = queries.write_chat_answer(
            self.user_id,
            "the answer",
            [_enhanced_slice(doc_a), _enhanced_slice(doc_b)],
            uuid.uuid4(),
        )

        returned = self._all(ReturnedDocument)
        self.assertEqual({r.document_id for r in returned}, {doc_a, doc_b})
        self.assertTrue(all(r.message_id == msg_id for r in returned))
        self.assertTrue(all(r.is_clicked is False for r in returned))


class TestWriteReturnedDocs(DataCollectionQueriesTestBase):
    def test_converts_scored_points_and_sets_clicked(self):
        msg_id, _ = self._add_message("search", datetime.now())
        doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
        scored = ScoredPoint(id=1, version=0, score=0.5, payload=_payload(doc_a))

        queries.write_returned_docs(
            msg_id, [scored, _enhanced_slice(doc_b)], is_clicked=True
        )

        returned = self._all(ReturnedDocument)
        self.assertEqual({r.document_id for r in returned}, {doc_a, doc_b})
        self.assertTrue(all(r.message_id == msg_id for r in returned))
        self.assertTrue(all(r.is_clicked is True for r in returned))


class TestWriteFiltersSearch(DataCollectionQueriesTestBase):
    def test_writes_sdg_and_source_filters(self):
        msg_id, _ = self._add_message("search", datetime.now())

        queries.write_filters_search(msg_id, [3, 7], ["corpus_a"])

        rows = self._all(FilterUsedInQuery)
        values = sorted((r.filter_type, r.filter_value) for r in rows)
        self.assertEqual(values, [("sdg", "3"), ("sdg", "7"), ("source", "corpus_a")])
        self.assertTrue(all(r.message_id == msg_id for r in rows))

    def test_none_filters_write_nothing(self):
        msg_id, _ = self._add_message("search", datetime.now())

        queries.write_filters_search(msg_id, None, None)

        self.assertEqual(self._all(FilterUsedInQuery), [])


class TestSyllabusLookups(DataCollectionQueriesTestBase):
    def test_last_syllabus_conversation_id_none_when_no_message(self):
        self.assertIsNone(queries.get_last_syllabus_conversation_id(self.user_id))

    def test_last_syllabus_conversation_id_returns_latest_creation(self):
        now = datetime.now()
        self._add_message("syllabus_creation", now - timedelta(hours=2))
        _, latest_conv = self._add_message(
            "syllabus_creation", now - timedelta(hours=1)
        )
        # newer messages of other features or other users are ignored
        self._add_message("syllabus_feedback", now)
        other_user = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=other_user))
            s.commit()
        self._add_message("syllabus_creation", now, user_id=other_user)

        self.assertEqual(
            queries.get_last_syllabus_conversation_id(self.user_id), latest_conv
        )

    def test_last_syllabus_id_none_when_no_message(self):
        self._add_message("chat", datetime.now())

        self.assertIsNone(queries.get_last_syllabus_id_for_user(self.user_id))

    def test_last_syllabus_id_includes_feedback_and_updates(self):
        now = datetime.now()
        self._add_message("syllabus_creation", now - timedelta(hours=3))
        update_id, _ = self._add_message(
            "syllabus_user_update", now - timedelta(hours=1)
        )
        self._add_message("syllabus_feedback", now - timedelta(hours=2))
        self._add_message("chat", now)

        self.assertEqual(queries.get_last_syllabus_id_for_user(self.user_id), update_id)

    def test_update_syllabus_retrieved_status_sets_flag(self):
        msg_id, _ = self._add_message("syllabus_creation", datetime.now())

        queries.update_syllabus_retrieved_status(msg_id)

        with self.s_maker() as s:
            self.assertTrue(s.get(ChatMessage, msg_id).is_retrieved_by_user)

    def test_update_syllabus_retrieved_status_unknown_id_is_noop(self):
        msg_id, _ = self._add_message("syllabus_creation", datetime.now())

        queries.update_syllabus_retrieved_status(uuid.uuid4())

        with self.s_maker() as s:
            self.assertFalse(s.get(ChatMessage, msg_id).is_retrieved_by_user)


class TestUpdateReturnedDocumentClick(DataCollectionQueriesTestBase):
    def test_marks_matching_document_clicked(self):
        msg_id, _ = self._add_message("search", datetime.now())
        other_msg_id, _ = self._add_message("search", datetime.now())
        doc_id = uuid.uuid4()
        queries.write_returned_docs(msg_id, [_enhanced_slice(doc_id)])
        queries.write_returned_docs(other_msg_id, [_enhanced_slice(doc_id)])

        queries.update_returned_document_click(doc_id, msg_id)

        clicked = {r.message_id: r.is_clicked for r in self._all(ReturnedDocument)}
        self.assertEqual(clicked, {msg_id: True, other_msg_id: False})

    def test_unknown_document_is_noop(self):
        msg_id, _ = self._add_message("search", datetime.now())
        queries.write_returned_docs(msg_id, [_enhanced_slice(uuid.uuid4())])

        queries.update_returned_document_click(uuid.uuid4(), msg_id)

        self.assertFalse(self._all(ReturnedDocument)[0].is_clicked)


class TestGetCurrentDataCollectionCampaign(DataCollectionQueriesTestBase):
    def _add_campaign(self, end_at: datetime, is_active: bool = True):
        campaign_id = uuid.uuid4()
        with self.s_maker() as s:
            s.add(
                DataCollectionCampaignManagement(
                    id=campaign_id, is_active=is_active, end_at=end_at
                )
            )
            s.commit()
        return campaign_id

    def test_no_campaign(self):
        self.assertIsNone(queries.get_current_data_collection_campaign())

    def test_only_ended_campaigns(self):
        self._add_campaign(datetime(2000, 1, 1))

        self.assertIsNone(queries.get_current_data_collection_campaign())

    def test_returns_earliest_ending_future_campaign(self):
        self._add_campaign(datetime(2000, 1, 1))
        self._add_campaign(datetime(2999, 1, 1))
        soonest = self._add_campaign(datetime(2998, 1, 1), is_active=False)

        campaign = queries.get_current_data_collection_campaign()

        self.assertIsNotNone(campaign)
        self.assertEqual(campaign.id, soonest)
        self.assertFalse(campaign.is_active)
