import unittest
from unittest import mock

from langchain.agents.middleware import ClearToolUsesEdit, SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.app.chat.services import prompts
from src.app.chat.services.abst_chat import (
    AbstractChat,
    _PersistClearedToolUses,
    _ReinforceHardConstraints,
)


class TestAbstractChat(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        mocked_client = mock.AsyncMock()
        self.chat = AbstractChat(client=mocked_client)

    @mock.patch("src.app.chat.services.abst_chat.ChatMistralAI")
    @mock.patch("src.app.chat.services.abst_chat.get_settings")
    @mock.patch("src.app.chat.services.abst_chat.create_agent")
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

        cleared = next(
            m for m in new_messages if getattr(m, "tool_call_id", None) == "1"
        )
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
