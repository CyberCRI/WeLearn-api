import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.app.core.langsmith import configure_langsmith_tracing

ENV_KEYS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_API_KEY",
)


def _settings(enabled=True, project=None, endpoint=None, api_key=None):
    return SimpleNamespace(
        LANGSMITH_TRACING_ENABLED=enabled,
        LANGSMITH_PROJECT=project,
        LANGSMITH_ENDPOINT=endpoint,
        LANGSMITH_API_KEY=api_key,
    )


@patch("src.app.core.langsmith.logger")
class ConfigureLangsmithTracingTests(TestCase):
    def setUp(self):
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        self._env.stop()

    def test_disabled_sets_flag_false_and_nothing_else(self, logger_mock):
        configure_langsmith_tracing(_settings(enabled=False, project="p"))

        self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")
        self.assertNotIn("LANGCHAIN_PROJECT", os.environ)
        logger_mock.warning.assert_not_called()

    def test_enabled_with_full_config_exports_env(self, logger_mock):
        configure_langsmith_tracing(
            _settings(project="proj", endpoint="https://ls.example", api_key="k")
        )

        self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
        self.assertEqual(os.environ["LANGCHAIN_PROJECT"], "proj")
        self.assertEqual(os.environ["LANGCHAIN_ENDPOINT"], "https://ls.example")
        self.assertEqual(os.environ["LANGSMITH_API_KEY"], "k")
        logger_mock.warning.assert_not_called()
        logger_mock.info.assert_called_with(
            "langsmith_tracing_enabled=true project=%s endpoint=%s",
            "proj",
            "https://ls.example",
        )

    def test_enabled_with_missing_config_warns_for_each_key(self, logger_mock):
        configure_langsmith_tracing(_settings())

        self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
        for key in ENV_KEYS[1:]:
            self.assertNotIn(key, os.environ)
        warnings = [c.args[0] for c in logger_mock.warning.call_args_list]
        self.assertEqual(len(warnings), 3)
        self.assertTrue(any("LANGSMITH_PROJECT" in w for w in warnings))
        self.assertTrue(any("LANGSMITH_ENDPOINT" in w for w in warnings))
        self.assertTrue(any("LANGSMITH_API_KEY" in w for w in warnings))
