import unittest
from unittest import mock

from src.app.shared.infra import chat_models


class TestChatModels(unittest.TestCase):
    @mock.patch("src.app.shared.infra.chat_models.ChatMistralAI")
    def test_build_mistral_chat_model_forwards_params(self, mock_chat_mistral):
        result = chat_models.build_mistral_chat_model(
            "fake_model", "fake_key", temperature=0.8
        )

        mock_chat_mistral.assert_called_once_with(
            model_name="fake_model", mistral_api_key="fake_key", temperature=0.8
        )
        self.assertIs(result, mock_chat_mistral.return_value)

    @mock.patch("src.app.shared.infra.chat_models.AzureAIOpenAIApiChatModel")
    def test_build_azure_chat_model_forwards_params(self, mock_azure_model):
        result = chat_models.build_azure_chat_model(
            "fake_model",
            "fake_key",
            "https://fake.endpoint",
            "2024-01-01",
            temperature=0.8,
        )

        mock_azure_model.assert_called_once_with(
            model="fake_model",
            credential="fake_key",
            endpoint="https://fake.endpoint",
            api_version="2024-01-01",
            use_responses_api=False,
            temperature=0.8,
        )
        self.assertIs(result, mock_azure_model.return_value)

    @mock.patch("src.app.shared.infra.chat_models.build_mistral_chat_model")
    def test_build_chat_model_defaults_to_mistral(self, mock_build_mistral):
        result = chat_models.build_chat_model("fake_model", api_key="fake_key")

        mock_build_mistral.assert_called_once_with("fake_model", "fake_key")
        self.assertIs(result, mock_build_mistral.return_value)

    def test_build_chat_model_requires_api_key_for_mistral(self):
        with self.assertRaises(ValueError):
            chat_models.build_chat_model("fake_model")

    @mock.patch("src.app.shared.infra.chat_models.build_azure_chat_model")
    def test_build_chat_model_routes_to_azure(self, mock_build_azure):
        result = chat_models.build_chat_model(
            "fake_model",
            api_key="fake_key",
            api_base="https://fake.endpoint",
            api_version="2024-01-01",
            is_azure_model=True,
        )

        mock_build_azure.assert_called_once_with(
            "fake_model", "fake_key", "https://fake.endpoint", "2024-01-01"
        )
        self.assertIs(result, mock_build_azure.return_value)

    def test_build_chat_model_requires_all_azure_params(self):
        with self.assertRaises(ValueError):
            chat_models.build_chat_model(
                "fake_model", api_key="fake_key", is_azure_model=True
            )
