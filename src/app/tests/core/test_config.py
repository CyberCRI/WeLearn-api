import os
import unittest
from unittest.mock import patch

from psycopg.conninfo import conninfo_to_dict

from src.app.core.config import Settings, settings
from src.app.shared.utils.dependencies import get_settings


class PostgresConninfoTests(unittest.TestCase):
    def _settings(self, **overrides):
        return get_settings().model_copy(update=overrides)

    def test_password_with_url_special_characters_round_trips(self):
        # "%Og" made psycopg fail with "invalid percent-encoded token" in a URL (B-12)
        password = "p%Og#:@/{x}<>$ 'q'\\z"
        settings = self._settings(
            PG_USER="welearn",
            PG_PASSWORD=password,
            PG_HOST="db.example",
            PG_PORT="5432",
            PG_DATABASE="welearn_dev",
        )

        params = conninfo_to_dict(settings.postgres_conninfo())

        self.assertEqual(
            params,
            {
                "host": "db.example",
                "port": "5432",
                "dbname": "welearn_dev",
                "user": "welearn",
                "password": password,
            },
        )

    def test_unset_values_are_left_out(self):
        settings = self._settings(
            PG_USER=None,
            PG_PASSWORD=None,
            PG_HOST=None,
            PG_PORT=None,
            PG_DATABASE="welearn_dev",
        )

        self.assertEqual(
            conninfo_to_dict(settings.postgres_conninfo()), {"dbname": "welearn_dev"}
        )


class EmbeddingModelsToPreloadTests(unittest.TestCase):
    def test_default_preloads_the_multilingual_model(self):
        default = Settings.model_fields["EMBEDDING_MODELS_TO_PRELOAD"].default
        self.assertEqual(default, ["granite-embedding-107m-multilingual"])

    def test_env_value_is_read_as_a_json_list(self):
        for raw, expected in (
            ("[]", []),
            ('["model-a", "model-b"]', ["model-a", "model-b"]),
        ):
            with (
                self.subTest(raw=raw),
                patch.dict(os.environ, {"EMBEDDING_MODELS_TO_PRELOAD": raw}),
            ):
                self.assertEqual(Settings().EMBEDDING_MODELS_TO_PRELOAD, expected)


class ApiVersionTests(unittest.TestCase):
    def test_default_is_dev(self):
        # run from the source, without the value CI bakes into the image
        self.assertEqual(Settings.model_fields["APP_VERSION"].default, "dev")

    def test_reports_the_version_from_the_env(self):
        with patch.dict(os.environ, {"APP_VERSION": "v0.8.0"}):
            settings = Settings()

        self.assertEqual(settings.get_api_version()["semver"], "v0.8.0")

    def test_root_route_and_openapi_info_report_it(self):
        from fastapi.testclient import TestClient

        from src.main import app

        # no `with`: the lifespan (models, database) is not needed here
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["semver"], settings.APP_VERSION)
        self.assertEqual(app.openapi()["info"]["version"], settings.APP_VERSION)
