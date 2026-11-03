import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.app.tutor.domain.models import ExtractorOutput, ExtractorOutputList
from src.app.tutor.services import extractor
from src.main import app

client = TestClient(app)


def _llm_output():
    return extractor._Extracts(
        extracts=[
            extractor._Extract(
                summary="A summary",
                themes=[extractor._Theme(theme="Climate", reason="Mentions CO2")],
            )
        ]
    )


class TestExtractFilesSummaries(unittest.IsolatedAsyncioTestCase):
    def _mock_model(self, output=None, side_effect=None):
        structured = mock.MagicMock()
        structured.ainvoke = mock.AsyncMock(
            return_value=output, side_effect=side_effect
        )
        model = mock.MagicMock()
        model.with_structured_output.return_value = structured
        return model, structured

    def test_build_extractor_messages_injects_lang_and_documents(self):
        messages = extractor.build_extractor_messages("doc A", "fr")

        self.assertEqual([role for role, _ in messages], ["system", "user"])
        self.assertIn("lang: fr", messages[0][1])
        self.assertIn("doc A", messages[1][1])

    async def test_returns_public_model(self):
        model, structured = self._mock_model(output=_llm_output())

        with mock.patch.object(extractor.tutor, "chat_model", model):
            result = await extractor.extract_files_summaries("doc A", "en")

        model.with_structured_output.assert_called_once_with(extractor._Extracts)
        structured.ainvoke.assert_awaited_once()
        self.assertEqual(
            result,
            ExtractorOutputList(
                extracts=[
                    ExtractorOutput(
                        summary="A summary",
                        themes=[{"theme": "Climate", "reason": "Mentions CO2"}],
                    )
                ]
            ),
        )

    async def test_initialises_model_when_missing(self):
        model, _ = self._mock_model(output=_llm_output())

        async def fake_init(_settings):
            extractor.tutor.chat_model = model

        with (
            mock.patch.object(extractor.tutor, "chat_model", None),
            mock.patch.object(
                extractor.tutor, "init_chat_model", side_effect=fake_init
            ) as init,
        ):
            await extractor.extract_files_summaries("doc A", "en")

        init.assert_awaited_once()

    async def test_unexpected_output_raises(self):
        model, _ = self._mock_model(output={"extracts": []})

        with mock.patch.object(extractor.tutor, "chat_model", model):
            with self.assertRaises(ValueError):
                await extractor.extract_files_summaries("doc A", "en")

    async def test_llm_error_propagates(self):
        model, _ = self._mock_model(side_effect=RuntimeError("boom"))

        with mock.patch.object(extractor.tutor, "chat_model", model):
            with self.assertRaises(RuntimeError):
                await extractor.extract_files_summaries("doc A", "en")


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class TestFilesContentRoute(unittest.TestCase):
    @mock.patch(
        "src.app.tutor.api.router.extract_files_summaries",
        new_callable=mock.AsyncMock,
    )
    @mock.patch(
        "src.app.tutor.api.router.get_files_content", new_callable=mock.AsyncMock
    )
    def test_files_content_returns_extracts(self, get_files_content, extract):
        get_files_content.return_value = ["doc A", "doc B"]
        extract.return_value = ExtractorOutputList(
            extracts=[
                ExtractorOutput(
                    summary="A summary",
                    themes=[{"theme": "Climate", "reason": "Mentions CO2"}],
                )
            ]
        )

        response = client.post(
            "/api/v1/tutor/files/content?lang=fr",
            headers={"X-API-Key": "test"},
            files=[("files", ("a.txt", b"doc A", "text/plain"))],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "extracts": [
                    {
                        "summary": "A summary",
                        "themes": [{"theme": "Climate", "reason": "Mentions CO2"}],
                    }
                ]
            },
        )
        extract.assert_awaited_once_with("doc A__DOCUMENT_SEPARATOR__doc B", "fr")
