import logging


def format_records(records: list[logging.LogRecord]) -> str:
    """Every record as a handler would print it, traceback included."""
    formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
    return "\n".join(formatter.format(record) for record in records)
