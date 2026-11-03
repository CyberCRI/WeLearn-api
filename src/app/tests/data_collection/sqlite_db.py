"""
In-memory sqlite database for testing the sync query modules.

The welearn_database models declare their ids and timestamps as Postgres
server defaults (gen_random_uuid(), NOW()), which sqlite stores as literal
strings. A before_flush hook fills those columns client-side instead, so the
real query functions can run unchanged against sqlite.
"""

import uuid
from datetime import datetime

import sqlalchemy
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from welearn_database.data.models import Base, DbSchemaEnum


def _fill_server_defaults(session, flush_context, instances):
    now = datetime.now()
    for obj in session.new:
        if hasattr(obj, "id") and obj.id is None:
            obj.id = uuid.uuid4()
        for column in ("created_at", "updated_at"):
            if hasattr(obj, column) and getattr(obj, column) is None:
                setattr(obj, column, now)


def make_sqlite_session_maker():
    """Return (engine, sessionmaker) bound to a fresh in-memory database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        for schema_name in DbSchemaEnum:  # type: ignore
            conn.execute(sqlalchemy.text(f"ATTACH ':memory:' AS {schema_name.value}"))
    Base.metadata.create_all(engine)

    s_maker = sessionmaker(bind=engine)
    event.listen(s_maker, "before_flush", _fill_server_defaults)
    return engine, s_maker
