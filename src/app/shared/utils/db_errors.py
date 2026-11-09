"""
Database errors are logged by type only (C-9).

psycopg and SQLAlchemy error messages can quote the connection string, password
included: a password psycopg cannot parse ends up in the "invalid connection
info" message, and SQLAlchemy repeats the driver message in its own. Their text,
and tracebacks (which end with that text), must therefore never reach the logs
or a response body. Log `describe_error(e)` instead of `e`.
"""

import logging
import traceback
from typing import Iterator

import psycopg
from sqlalchemy import exc as sa_exc

DB_ERRORS: tuple[type[Exception], ...] = (psycopg.Error, sa_exc.DBAPIError)

# errors meaning the database cannot be reached or the connection was lost
DB_CONNECTION_ERRORS: tuple[type[Exception], ...] = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
    sa_exc.OperationalError,
    sa_exc.InterfaceError,
)

REDACTED = "database error, details not logged"


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """`exc`, then its causes and contexts, each once."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def involves_db_error(exc: BaseException) -> bool:
    """True when `exc` is a database error or was raised while handling one."""
    return any(isinstance(e, DB_ERRORS) for e in _exception_chain(exc))


def is_db_connection_error(exc: BaseException) -> bool:
    return isinstance(exc, DB_CONNECTION_ERRORS)


def _type_name(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def _sqlstate(exc: BaseException) -> str | None:
    """The SQLSTATE code, safe to log, from a psycopg (3 or 2) or SQLAlchemy error."""
    for candidate in (exc, getattr(exc, "orig", None)):
        code = getattr(candidate, "sqlstate", None) or getattr(
            candidate, "pgcode", None
        )
        if isinstance(code, str):
            return code
    return None


def describe_error(exc: BaseException) -> str:
    """
    A loggable description of `exc`: its message for ordinary errors, its type
    (with the driver error type and SQLSTATE when known) for database errors.
    """
    if not involves_db_error(exc):
        return str(exc)

    names = [_type_name(exc)]
    orig = getattr(exc, "orig", None)
    if isinstance(orig, BaseException):
        names.append(_type_name(orig))
    sqlstate = _sqlstate(exc)
    if sqlstate:
        names.append(f"sqlstate={sqlstate}")
    return f"{' '.join(names)} ({REDACTED})"


def describe_traceback(exc: BaseException) -> str:
    """The stack frames of `exc`, without any exception message."""
    return "".join(traceback.format_tb(exc.__traceback__))


class DbErrorRedactionFilter(logging.Filter):
    """
    Safety net for log calls that pass a database error as is: replaces it with
    `describe_error` in the arguments, and drops the traceback text (keeping the
    frames) when a record's `exc_info` involves one. Covers `logger.exception`
    and the "Exception in ASGI application" traceback uvicorn logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(self._redact(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: self._redact(v) for k, v in record.args.items()}

        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None and involves_db_error(exc):
            record.exc_info = None
            record.exc_text = None
            record.msg = (
                f"{record.getMessage()} error={describe_error(exc)}\n"
                f"Traceback (message redacted):\n{describe_traceback(exc)}"
            )
            record.args = None
        return True

    @staticmethod
    def _redact(arg: object) -> object:
        if isinstance(arg, BaseException) and involves_db_error(arg):
            return describe_error(arg)
        return arg


def install_db_error_redaction(target: logging.Logger | logging.Handler) -> None:
    """Add a `DbErrorRedactionFilter` to `target`, once."""
    if not any(isinstance(f, DbErrorRedactionFilter) for f in target.filters):
        target.addFilter(DbErrorRedactionFilter())
