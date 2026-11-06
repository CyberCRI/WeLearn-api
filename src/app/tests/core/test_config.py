import unittest

from psycopg.conninfo import conninfo_to_dict

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
