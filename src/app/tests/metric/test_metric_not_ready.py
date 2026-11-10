import unittest
from unittest import mock

import psycopg.errors
from fastapi.testclient import TestClient
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from src.app.core.config import settings
from src.app.metric.infra import queries
from src.app.metric.infra.queries import get_document_qty_table_info_sync
from src.app.shared.domain.exceptions import MetricsNotReadyError
from src.main import app

URL = f"{settings.API_V1_STR}/metric/nb_docs_info_per_corpus"
HEADERS = {"X-API-Key": "test"}
QUERY = "src.app.metric.infra.queries._query_document_qty_table_info"
IN_QDRANT_VIEW = "document_related.qty_document_in_qdrant_per_corpus"
PER_CORPUS_VIEW = "document_related.qty_document_per_corpus"


def _not_populated(view: str = "qty_document_in_qdrant_per_corpus") -> DBAPIError:
    # what SQLAlchemy raises with psycopg 3 when the view was never refreshed
    orig = psycopg.errors.ObjectNotInPrerequisiteState(
        f'materialized view "{view}" has not been populated'
    )
    return OperationalError("SELECT ...", {}, orig)


class _Psycopg2Error(Exception):
    # psycopg2 errors carry the SQLSTATE in `pgcode`, set by the driver
    def __init__(self, message: str, pgcode: str):
        super().__init__(message)
        self.pgcode = pgcode


class DocumentQtyNotPopulatedTests(unittest.TestCase):
    def test_unpopulated_view_raises_metrics_not_ready_and_logs_refresh_hint(self):
        with (
            mock.patch(QUERY, side_effect=_not_populated()),
            self.assertLogs(queries.logger.name, level="ERROR") as logs,
            self.assertRaises(MetricsNotReadyError),
        ):
            get_document_qty_table_info_sync()

        self.assertEqual(len(logs.records), 1)
        line = logs.records[0].getMessage()
        self.assertIn(IN_QDRANT_VIEW, line)
        self.assertIn(f"REFRESH MATERIALIZED VIEW {IN_QDRANT_VIEW};", line)
        self.assertNotIn(PER_CORPUS_VIEW, line)
        self.assertIsNone(logs.records[0].exc_info)

    def test_psycopg2_error_is_recognised_by_pgcode(self):
        orig = _Psycopg2Error(
            'materialized view "qty_document_per_corpus" has not been populated',
            "55000",
        )
        with (
            mock.patch(QUERY, side_effect=OperationalError("SELECT ...", {}, orig)),
            self.assertLogs(queries.logger.name, level="ERROR") as logs,
            self.assertRaises(MetricsNotReadyError),
        ):
            get_document_qty_table_info_sync()

        self.assertIn(f"REFRESH MATERIALIZED VIEW {PER_CORPUS_VIEW};", logs.output[0])

    def test_unnamed_view_lists_every_metric_view(self):
        orig = _Psycopg2Error("not populated", "55000")
        with (
            mock.patch(QUERY, side_effect=OperationalError("SELECT ...", {}, orig)),
            self.assertLogs(queries.logger.name, level="ERROR") as logs,
            self.assertRaises(MetricsNotReadyError),
        ):
            get_document_qty_table_info_sync()

        self.assertIn(IN_QDRANT_VIEW, logs.output[0])
        self.assertIn(PER_CORPUS_VIEW, logs.output[0])

    def test_other_db_errors_are_re_raised(self):
        error = ProgrammingError(
            "SELECT ...", {}, psycopg.errors.UndefinedTable("relation missing")
        )
        with mock.patch(QUERY, side_effect=error):
            with self.assertRaises(ProgrammingError) as ctx:
                get_document_qty_table_info_sync()
        self.assertIs(ctx.exception, error)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class NbDocsInfoNotPopulatedRouteTests(unittest.TestCase):
    def test_unpopulated_view_returns_503_without_sql(self):
        client = TestClient(app)
        with (
            mock.patch(QUERY, side_effect=_not_populated()),
            self.assertLogs(queries.logger.name, level="ERROR") as logs,
        ):
            response = client.get(URL, headers=HEADERS)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "message": "Metrics are not available yet",
                    "code": "METRICS_NOT_READY",
                }
            },
        )
        self.assertNotIn("SELECT", response.text)
        self.assertNotIn("materialized", response.text)
        self.assertIn(f"REFRESH MATERIALIZED VIEW {IN_QDRANT_VIEW};", logs.output[0])

    def test_other_db_error_is_not_metrics_not_ready(self):
        client = TestClient(app, raise_server_exceptions=False)
        error = OperationalError(
            "SELECT ...", {}, psycopg.errors.QueryCanceled("canceling statement")
        )
        with mock.patch(QUERY, side_effect=error):
            response = client.get(URL, headers=HEADERS)

        # mapped by the global database handler (C-9), not by B-13
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "DATABASE_UNAVAILABLE")
        self.assertNotIn("canceling statement", response.text)
