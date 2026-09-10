import unittest
from unittest import mock

from src.app.shared.domain.exceptions import LanguageNotSupportedError
from src.app.shared.infra.abst_chat import AbstractChat


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

    @mock.patch("src.app.shared.infra.abst_chat.create_agent")
    @mock.patch("src.app.shared.infra.abst_chat.build_chat_model")
    async def test_create_agent_adds_summarization_middleware(
        self, mock_build_chat_model, mock_create_agent
    ):
        mocked_model = mock.Mock()
        mocked_model._llm_type = "mistral-chat"  # noqa: SLF001
        mock_build_chat_model.return_value = mocked_model
        mock_create_agent.return_value = object()

        await self.chat._create_agent(memory=None)

        middleware = mock_create_agent.call_args.kwargs["middleware"]
        assert len(middleware) == 1
        assert middleware[0].__class__.__name__ == "SummarizationMiddleware"

    async def test_judge_source_grounding_skips_llm_when_no_citations(self):
        verdict = await self.chat.judge_source_grounding("Plain answer, no refs.", [])

        self.assertTrue(verdict.compliant)
        self.assertEqual(verdict.unsupported_citations, [])

    async def test_judge_source_grounding_skips_llm_when_answer_empty(self):
        verdict = await self.chat.judge_source_grounding("", None)

        self.assertTrue(verdict.compliant)

    @mock.patch("src.app.shared.infra.abst_chat.build_chat_model")
    async def test_judge_source_grounding_calls_structured_llm_when_citations_present(
        self, mock_build_chat_model
    ):
        from src.app.models.chat import SourceGroundingVerdict

        expected_verdict = SourceGroundingVerdict(
            compliant=False,
            unsupported_citations=["[Doc 3]"],
            reasoning="Doc 3 was never retrieved.",
        )
        structured_model = mock.Mock()
        structured_model.ainvoke = mock.AsyncMock(return_value=expected_verdict)
        judge_model = mock.Mock()
        judge_model.with_structured_output = mock.Mock(return_value=structured_model)
        mock_build_chat_model.return_value = judge_model

        answer = 'See <a href="https://example.com">[Doc 3]</a> for details.'
        verdict = await self.chat.judge_source_grounding(answer, [])

        self.assertIs(verdict, expected_verdict)
        judge_model.with_structured_output.assert_called_once_with(
            SourceGroundingVerdict
        )
        structured_model.ainvoke.assert_awaited_once()

    @mock.patch("src.app.shared.infra.abst_chat.build_chat_model")
    async def test_judge_source_grounding_reuses_cached_model(
        self, mock_build_chat_model
    ):
        from src.app.models.chat import SourceGroundingVerdict

        structured_model = mock.Mock()
        structured_model.ainvoke = mock.AsyncMock(
            return_value=SourceGroundingVerdict(compliant=True)
        )
        judge_model = mock.Mock()
        judge_model.with_structured_output = mock.Mock(return_value=structured_model)
        mock_build_chat_model.return_value = judge_model

        await self.chat.judge_source_grounding("Cites [Doc 1] here.", [])
        await self.chat.judge_source_grounding("Cites [Doc 2] here.", [])

        mock_build_chat_model.assert_called_once()
