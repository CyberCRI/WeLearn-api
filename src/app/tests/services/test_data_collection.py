import unittest
import uuid
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from src.app.services.data_collection import DataCollection, _cache


async def fake_run_in_threadpool(func, *args, **kwargs):
    return func(*args, **kwargs)


class TestDataCollectionCampaignState(unittest.TestCase):
    def setUp(self):
        _cache["is_campaign_active"] = None
        _cache["expires"] = None

    @patch("src.app.services.data_collection.get_current_data_collection_campaign")
    def test_campaign_active(self, mock_get_campaign):
        mock_campaign = MagicMock()
        mock_campaign.is_active = True
        mock_get_campaign.return_value = mock_campaign

        dc = DataCollection(origin="workshop.example.com")

        self.assertTrue(dc.should_collect)

    @patch("src.app.services.data_collection.get_current_data_collection_campaign")
    def test_campaign_inactive(self, mock_get_campaign):
        mock_campaign = MagicMock()
        mock_campaign.is_active = False
        mock_get_campaign.return_value = mock_campaign

        dc = DataCollection(origin="workshop.example.com")

        self.assertFalse(dc.should_collect)

    @patch("src.app.services.data_collection.get_current_data_collection_campaign")
    def test_non_workshop_origin(self, mock_get_campaign):
        mock_campaign = MagicMock()
        mock_campaign.is_active = True
        mock_get_campaign.return_value = mock_campaign

        dc = DataCollection(origin="example.com")

        self.assertFalse(dc.should_collect)


class TestRegisterChatData(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        _cache["is_campaign_active"] = True
        _cache["expires"] = None

    @patch(
        "src.app.services.data_collection.run_in_threadpool",
        side_effect=fake_run_in_threadpool,
    )
    @patch("src.app.services.data_collection.write_chat_answer")
    @patch("src.app.services.data_collection.write_user_query")
    @patch("src.app.services.data_collection.get_user_from_session_id")
    @patch("src.app.services.data_collection.get_current_data_collection_campaign")
    async def test_register_chat_data_success(
        self, mock_campaign, mock_get_user, mock_write_query, mock_write_answer, _
    ):
        mock_campaign.return_value = MagicMock(is_active=True)

        user_id = uuid.uuid4()
        conversation_id = uuid.uuid4()
        message_id = uuid.uuid4()

        mock_get_user.return_value = user_id
        mock_write_query.return_value = conversation_id
        mock_write_answer.return_value = message_id

        dc = DataCollection(origin="workshop.example.com")

        result = await dc.register_chat_data(
            session_id=str(uuid.uuid4()),
            user_query="hello",
            conversation_id=None,
            answer_content="hi",
            sources=[],
        )

        self.assertEqual(result, (conversation_id, message_id))

    @patch(
        "src.app.services.data_collection.run_in_threadpool",
        side_effect=fake_run_in_threadpool,
    )
    @patch("src.app.services.data_collection.write_chat_answer")
    @patch("src.app.services.data_collection.write_user_query")
    @patch("src.app.services.data_collection.get_user_from_session_id")
    @patch("src.app.services.data_collection.get_current_data_collection_campaign")
    async def test_register_chat_data_no_session(self, *args):
        dc = DataCollection(origin="workshop.example.com")

        with self.assertRaises(HTTPException) as ctx:
            await dc.register_chat_data(
                session_id=None,
                user_query="hello",
                conversation_id=None,
                answer_content="hi",
                sources=[],
            )

        self.assertEqual(ctx.exception.status_code, 401)

    @patch(
        "src.app.services.data_collection.run_in_threadpool",
        side_effect=fake_run_in_threadpool,
    )
    @patch(
        "src.app.services.data_collection.get_user_from_session_id", return_value=None
    )
    @patch("src.app.services.data_collection.get_current_data_collection_campaign")
    async def test_register_chat_data_user_not_found(self, mock_campaign, _, __):
        mock_campaign.return_value = MagicMock(is_active=True)

        dc = DataCollection(origin="workshop.example.com")

        with self.assertRaises(HTTPException) as ctx:
            await dc.register_chat_data(
                session_id=str(uuid.uuid4()),
                user_query="hello",
                conversation_id=None,
                answer_content="hi",
                sources=[],
            )

        self.assertEqual(ctx.exception.status_code, 401)


class TestRegisterDocumentClick(unittest.IsolatedAsyncioTestCase):

    @patch(
        "src.app.services.data_collection.run_in_threadpool",
        side_effect=fake_run_in_threadpool,
    )
    @patch("src.app.services.data_collection.update_returned_document_click")
    @patch("src.app.services.data_collection.get_current_data_collection_campaign")
    async def test_register_document_click(self, mock_campaign, mock_update, _):
        mock_campaign.return_value = MagicMock(is_active=True)

        dc = DataCollection(origin="workshop.example.com")

        doc_id = uuid.uuid4()
        message_id = uuid.uuid4()

        await dc.register_document_click(doc_id, message_id)

        mock_update.assert_called_once_with(doc_id, message_id)
