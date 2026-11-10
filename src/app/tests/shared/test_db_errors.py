import logging
import unittest
import uuid
from itertools import islice
from unittest import TestCase, mock

import psycopg
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError as SAOperationalError
from sqlalchemy.exc import ProgrammingError as SAProgrammingError

from src.app.shared.api.errors import register_exception_handlers
from src.app.shared.utils import db_errors
from src.app.shared.utils.db_errors import (
    DbErrorRedactionFilter,
    _exception_chain,
    describe_error,
    install_db_error_redaction,
    involves_db_error,
)
from src.app.shared.utils.decorators import log_time_and_error
from src.app.shared.utils.logger import logger as utils_logger
from src.app.tests.shared.log_helpers import format_records

SECRET = "S3cr3t%#"
# what psycopg says when it cannot parse a conninfo holding that password
PARSE_ERROR = f'invalid percent-encoded token: "{SECRET}"'
CONNINFO = f"host=db user=welearn password={SECRET}"


def _raise(exc: BaseException):
    raise exc


class DescribeErrorTests(TestCase):
    def test_psycopg_error_is_described_by_type_only(self):
        described = describe_error(psycopg.ProgrammingError(PARSE_ERROR))

        self.assertIn("psycopg.ProgrammingError", described)
        self.assertIn(db_errors.REDACTED, described)
        self.assertNotIn(SECRET, described)

    def test_sqlalchemy_error_names_the_driver_error_and_sqlstate(self):
        orig = psycopg.OperationalError(f"connection failed {CONNINFO}")
        orig.pgcode = "08001"  # type: ignore[attr-defined]  # as psycopg2 sets it
        exc = SAOperationalError("SELECT 1", {}, orig)

        described = describe_error(exc)

        self.assertIn("sqlalchemy.exc.OperationalError", described)
        self.assertIn("psycopg.OperationalError", described)
        self.assertIn("sqlstate=08001", described)
        self.assertNotIn(SECRET, described)

    def test_error_raised_while_handling_a_database_error_is_redacted(self):
        try:
            try:
                raise psycopg.OperationalError(CONNINFO)
            except psycopg.Error as e:
                raise RuntimeError(f"wrapped: {e}")
        except RuntimeError as wrapped:
            exc = wrapped

        self.assertTrue(involves_db_error(exc))
        self.assertNotIn(SECRET, describe_error(exc))
        self.assertIn("RuntimeError", describe_error(exc))

    def test_other_errors_keep_their_message(self):
        self.assertEqual(describe_error(ValueError("qdrant down")), "qdrant down")
        self.assertFalse(involves_db_error(ValueError("qdrant down")))


class ExceptionChainTests(TestCase):
    """A cyclic chain must not make the walk (run in every log filter) loop forever."""

    @staticmethod
    def _context_cycle() -> tuple[BaseException, BaseException]:
        a, b = RuntimeError("a"), ValueError("b")
        a.__context__ = b
        b.__context__ = a
        return a, b

    def test_cyclic_chain_yields_each_exception_once(self):
        a, b = self._context_cycle()

        # islice bounds the walk: without the cycle guard this fails instead of hanging
        self.assertEqual(list(islice(_exception_chain(a), 10)), [a, b])

    def test_cyclic_chain_without_db_error_is_not_a_db_error(self):
        a, _ = self._context_cycle()

        self.assertFalse(involves_db_error(a))
        self.assertEqual(describe_error(a), "a")

    def test_cyclic_chain_through_cause_with_db_error_is_detected(self):
        a = RuntimeError("a")
        db = psycopg.OperationalError(CONNINFO)
        c = ValueError("c")
        a.__cause__ = db
        db.__cause__ = c
        c.__cause__ = a

        self.assertEqual(list(islice(_exception_chain(a), 10)), [a, db, c])
        self.assertTrue(involves_db_error(a))
        self.assertNotIn(SECRET, describe_error(a))


class RedactionFilterTests(TestCase):
    def setUp(self):
        self.log = logging.getLogger("tests.db_errors.redaction")
        self.log.setLevel(logging.DEBUG)
        install_db_error_redaction(self.log)

    def tearDown(self):
        self.log.filters.clear()

    def test_database_error_argument_is_replaced_by_its_description(self):
        with self.assertLogs(self.log, logging.DEBUG) as logs:
            self.log.error("failed: %s", psycopg.ProgrammingError(PARSE_ERROR))
            self.log.error(
                "failed: %(e)s", {"e": psycopg.ProgrammingError(PARSE_ERROR)}
            )

        output = format_records(logs.records)
        self.assertNotIn(SECRET, output)
        self.assertIn("psycopg.ProgrammingError", output)

    def test_traceback_of_a_database_error_keeps_frames_without_message(self):
        with self.assertLogs(self.log, logging.DEBUG) as logs:
            try:
                _raise(psycopg.ProgrammingError(PARSE_ERROR))
            except psycopg.Error:
                self.log.exception("Exception in ASGI application")

        output = format_records(logs.records)
        self.assertNotIn(SECRET, output)
        self.assertIn("Exception in ASGI application", output)
        self.assertIn("psycopg.ProgrammingError", output)
        self.assertIn("_raise", output)  # the frames are kept
        self.assertIsNone(logs.records[0].exc_info)

    def test_other_errors_are_left_untouched(self):
        with self.assertLogs(self.log, logging.DEBUG) as logs:
            try:
                raise ValueError("qdrant down")
            except ValueError as e:
                self.log.exception("failed: %s", e)

        self.assertIsNotNone(logs.records[0].exc_info)
        self.assertIn("qdrant down", format_records(logs.records))

    def test_installed_once(self):
        install_db_error_redaction(self.log)

        filters = [f for f in self.log.filters if isinstance(f, DbErrorRedactionFilter)]
        self.assertEqual(len(filters), 1)

    def test_project_and_uvicorn_loggers_have_the_filter(self):
        import src.main  # noqa: F401  installs it on uvicorn.error

        for log in (
            utils_logger("tests.db_errors.project"),
            logging.getLogger("uvicorn.error"),
        ):
            with self.subTest(logger=log.name):
                self.assertTrue(
                    any(isinstance(f, DbErrorRedactionFilter) for f in log.filters)
                )


class LogTimeAndErrorTests(TestCase):
    def test_database_error_is_logged_without_its_message(self):
        @log_time_and_error
        def query():
            raise SAOperationalError("SELECT 1", {}, Exception(CONNINFO))

        with self.assertLogs("src.app.shared.utils.decorators", "ERROR") as logs:
            with self.assertRaises(SAOperationalError):
                query()

        output = format_records(logs.records)
        self.assertNotIn(SECRET, output)
        self.assertIn("sqlalchemy.exc.OperationalError", output)


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    errors = {
        "psycopg_connect": lambda: psycopg.OperationalError(CONNINFO),
        "psycopg_interface": lambda: psycopg.InterfaceError(CONNINFO),
        "psycopg_parse": lambda: psycopg.ProgrammingError(PARSE_ERROR),
        "sqlalchemy_connect": lambda: SAOperationalError(
            "SELECT 1", {}, Exception(CONNINFO)
        ),
        "sqlalchemy_query": lambda: SAProgrammingError(
            "SELECT 1", {}, Exception(CONNINFO)
        ),
    }

    @app.get("/raise/{name}")
    async def raise_error(name: str):
        raise errors[name]()

    return app


class DatabaseErrorHandlerTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(_build_app())

    def test_database_errors_answer_a_generic_error_and_log_no_message(self):
        cases = {
            "psycopg_connect": (503, "DATABASE_UNAVAILABLE"),
            "psycopg_interface": (503, "DATABASE_UNAVAILABLE"),
            "psycopg_parse": (500, "DATABASE_ERROR"),
            "sqlalchemy_connect": (503, "DATABASE_UNAVAILABLE"),
            "sqlalchemy_query": (500, "DATABASE_ERROR"),
        }
        for name, (status_code, code) in cases.items():
            with self.subTest(name=name):
                with self.assertLogs(level=logging.DEBUG) as logs:
                    response = self.client.get(f"/raise/{name}")

                self.assertEqual(response.status_code, status_code)
                self.assertEqual(response.json()["detail"]["code"], code)
                self.assertNotIn(SECRET, response.text)
                output = format_records(logs.records)
                self.assertNotIn(SECRET, output)
                self.assertIn("Database error", output)
                self.assertIn("raise_error", output)  # stack frames kept


class MonitorRequestsTests(TestCase):
    @mock.patch(
        "src.app.shared.infra.security.check_api_key_sync",
        new=mock.MagicMock(return_value=True),
    )
    def test_failed_endpoint_registration_logs_no_dsn(self):
        from src.main import app

        with (
            mock.patch(
                "src.app.shared.api.monitor_requests.register_endpoint",
                side_effect=SAOperationalError("INSERT", {}, Exception(CONNINFO)),
            ),
            self.assertLogs("src.app.shared.api.monitor_requests", "ERROR") as logs,
        ):
            TestClient(app).get(
                "/api/v1/unknown-route",
                headers={"X-API-Key": "test", "X-Session-Id": str(uuid.uuid4())},
            )

        output = format_records(logs.records)
        self.assertNotIn(SECRET, output)
        self.assertIn("sqlalchemy.exc.OperationalError", output)


if __name__ == "__main__":
    unittest.main()
