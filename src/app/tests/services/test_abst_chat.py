import unittest
from unittest import mock

from langchain.agents.middleware import ClearToolUsesEdit, SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.app.services import prompts
from src.app.shared.domain.exceptions import LanguageNotSupportedError
from src.app.shared.infra.abst_chat import (
    AbstractChat,
    _PersistClearedToolUses,
    _ReinforceHardConstraints,
)


class TestAbstractChat(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        mocked_client = mock.AsyncMock()
        self.chat = AbstractChat(client=mocked_client)

    @mock.patch("src.app.shared.infra.abst_chat.detect_language_from_entry")
    async def test_lang_error_helper(self, mock_detect_lang):
        self.chat._detect_lang_with_llm = mock.AsyncMock()

        mock_detect_lang.side_effect = LanguageNotSupportedError
        await self.chat._detect_language("fake message")
        self.chat._detect_lang_with_llm.assert_called_once()

    @mock.patch(
        "src.app.shared.infra.abst_chat.detect_language_from_entry", return_value="en"
    )
    async def test_lang_ok(self, mock_detect_lang):
        lang = await self.chat._detect_language("fake message")
        assert lang == {"ISO_CODE": "en"}

    @mock.patch(
        "src.app.shared.infra.abst_chat.detect_language_from_entry",
        side_effect=LanguageNotSupportedError,
    )
    async def test_lang_not_supported(self, mock_detect_lang):
        mocked_chat = {"ISO_CODE": "pt"}
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)

        mocked_chat = "not json format"
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        with self.assertRaises(ValueError):
            await self.chat._detect_language("fake message")

    @mock.patch(
        "src.app.shared.infra.abst_chat.detect_language_from_entry",
        side_effect=LanguageNotSupportedError,
    )
    async def test_lang_supported(self, mock_detect_lang):
        mocked_chat = {"ISO_CODE": "en"}
        self.chat.chat_client.completion = mock.AsyncMock(
            return_value={"ISO_CODE": "en"}
        )
        assert await self.chat._detect_language("fake message") == {"ISO_CODE": "en"}

        mocked_chat = {"ISO_CODE": "fr"}
        self.chat.chat_client.completion = mock.AsyncMock(return_value=mocked_chat)
        assert await self.chat._detect_language("fake message") == {"ISO_CODE": "fr"}

    async def test_get_new_questions(self):
        with mock.patch.object(
            self.chat, "_detect_language", new_callable=mock.AsyncMock
        ) as mock_detect_lang:
            mock_detect_lang.return_value = {"ISO_CODE": "en"}
            self.chat.chat_client.completion = mock.AsyncMock(
                return_value="%%Question 1?%% Question 2?%%",
            )
            new_questions = await self.chat.get_new_questions(
                "this is the user query", []
            )

            mock_detect_lang.assert_called_with("this is the user query")
            assert new_questions == {"NEW_QUESTIONS": ["Question 1?", "Question 2?"]}

    async def test_chat_message(self):
        with mock.patch.object(
            self.chat, "_detect_language", new_callable=mock.AsyncMock
        ) as mock_detect_lang:
            self.chat.chat_client.completion = mock.AsyncMock()
            self.chat.chat_client.completion_stream = mock.AsyncMock()

            await self.chat.chat_message(
                query="this is a query",
                history=[],
                docs=[],
                subject="default",
                streamed_ans=True,
            )

            mock_detect_lang.assert_called_with("this is a query")
            self.chat.chat_client.completion.assert_not_called()
            self.chat.chat_client.completion_stream.assert_called_once()

    @mock.patch("src.app.shared.infra.abst_chat.ChatMistralAI")
    @mock.patch("src.app.shared.infra.abst_chat.get_settings")
    @mock.patch("src.app.shared.infra.abst_chat.create_agent")
    async def test_create_agent_adds_middleware_in_order(
        self, mock_create_agent, mock_get_settings, mock_chat_mistral_ai
    ):
        mocked_model = mock.Mock()
        mocked_model._llm_type = "mistral-chat"  # noqa: SLF001
        mock_chat_mistral_ai.return_value = mocked_model
        mock_create_agent.return_value = object()

        await self.chat._create_agent()

        _, kwargs = mock_create_agent.call_args
        middleware = kwargs["middleware"]
        self.assertEqual(len(middleware), 3)

        clear_uses, summarization, reinforcement = middleware

        self.assertIsInstance(clear_uses, _PersistClearedToolUses)
        edit = clear_uses._edit
        self.assertIsInstance(edit, ClearToolUsesEdit)
        self.assertEqual(edit.keep, 1)
        self.assertEqual(edit.trigger, 0)
        self.assertEqual(edit.clear_at_least, 0)

        self.assertIsInstance(summarization, SummarizationMiddleware)
        self.assertEqual(summarization.trigger, ("tokens", 32000))

        self.assertIsInstance(reinforcement, _ReinforceHardConstraints)

    async def test_persist_cleared_tool_uses_evicts_older_results_only(self):
        edit = ClearToolUsesEdit(
            trigger=0, clear_at_least=0, keep=1, placeholder="[cleared]"
        )
        middleware = _PersistClearedToolUses(edit)

        state = {
            "messages": [
                HumanMessage(content="q1"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "1",
                            "name": "get_resources_about_sustainability",
                            "args": {},
                        }
                    ],
                ),
                ToolMessage(
                    content="old result", tool_call_id="1", artifact=[{"id": "old"}]
                ),
                HumanMessage(content="q2"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "2",
                            "name": "get_resources_about_sustainability",
                            "args": {},
                        }
                    ],
                ),
                ToolMessage(
                    content="new result", tool_call_id="2", artifact=[{"id": "new"}]
                ),
            ]
        }

        result = middleware.before_model(state, runtime=mock.Mock())
        new_messages = result["messages"][1:]  # [0] is the RemoveMessage marker

        cleared = next(m for m in new_messages if getattr(m, "tool_call_id", None) == "1")
        kept = next(m for m in new_messages if getattr(m, "tool_call_id", None) == "2")

        self.assertEqual(cleared.content, "[cleared]")
        self.assertIsNone(cleared.artifact)
        self.assertEqual(kept.artifact, [{"id": "new"}])

    def test_reinforce_hard_constraints_appends_reminder_to_last_human_message(self):
        middleware = _ReinforceHardConstraints()

        result = middleware._with_reminder([HumanMessage(content="hello")])

        self.assertTrue(result[-1].content.startswith("hello"))
        self.assertIn(prompts.AGENT_REMINDER_PROMPT, result[-1].content)

    def test_reinforce_hard_constraints_leaves_non_human_last_message_alone(self):
        middleware = _ReinforceHardConstraints()
        messages = [HumanMessage(content="hello"), AIMessage(content="hi")]

        result = middleware._with_reminder(messages)

        self.assertEqual(result[-1].content, "hi")
