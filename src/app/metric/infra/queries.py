# src/app/metric/infra/queries.py
from sqlalchemy import func
from sqlalchemy.exc import DBAPIError
from welearn_database.data.models import (
    Corpus,
    QtyDocumentInQdrantPerCorpus,
    QtyDocumentPerCorpus,
)

from src.app.shared.domain.exceptions import MetricsNotReadyError
from src.app.shared.infra.db import session_maker
from src.app.shared.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

# SQLSTATE object_not_in_prerequisite_state: what Postgres raises when a query
# reads a materialized view that has never been refreshed
OBJECT_NOT_IN_PREREQUISITE_STATE = "55000"

METRIC_VIEWS = (
    QtyDocumentInQdrantPerCorpus.__table__.fullname,
    QtyDocumentPerCorpus.__table__.fullname,
)


def _sqlstate(exc: DBAPIError) -> str | None:
    # psycopg 3 exposes `sqlstate`, psycopg2 `pgcode`
    orig = exc.orig
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


def _unpopulated_views(exc: DBAPIError) -> list[str]:
    # Postgres names the view, unqualified and quoted:
    # materialized view "qty_document_in_qdrant_per_corpus" has not been populated
    message = str(exc.orig)
    named = [v for v in METRIC_VIEWS if f'"{v.rsplit(".", 1)[-1]}"' in message]
    return named or list(METRIC_VIEWS)


def get_document_qty_table_info_sync() -> list[tuple[Corpus, int, int]]:
    """
    Rows of (corpus, documents in Qdrant, documents), sub-corpora summed into
    their parent. Raises MetricsNotReadyError when a materialized view the
    query reads has not been populated.
    """
    try:
        return _query_document_qty_table_info()
    except DBAPIError as e:
        if _sqlstate(e) != OBJECT_NOT_IN_PREREQUISITE_STATE:
            raise
        views = _unpopulated_views(e)
        logger.error(
            "Materialized view not populated: %s. Metrics stay unavailable (503)"
            " until it is refreshed: %s",
            ", ".join(views),
            " ".join(f"REFRESH MATERIALIZED VIEW {v};" for v in views),
        )
        raise MetricsNotReadyError() from e


def _query_document_qty_table_info() -> list[tuple[Corpus, int, int]]:
    group_id = func.coalesce(Corpus.parent_corpus_id, Corpus.id)

    with session_maker() as s:
        totals = (
            s.query(
                group_id.label("group_id"),
                func.coalesce(func.sum(QtyDocumentInQdrantPerCorpus.count), 0).label(
                    "qty_in_qdrant"
                ),
                func.coalesce(func.sum(QtyDocumentPerCorpus.count), 0).label(
                    "qty_documents"
                ),
            )
            .select_from(Corpus)
            .outerjoin(
                QtyDocumentInQdrantPerCorpus,
                Corpus.source_name == QtyDocumentInQdrantPerCorpus.source_name,
            )
            .outerjoin(
                QtyDocumentPerCorpus,
                Corpus.source_name == QtyDocumentPerCorpus.source_name,
            )
            .group_by(group_id)
            .subquery()
        )

        return (
            s.query(Corpus, totals.c.qty_in_qdrant, totals.c.qty_documents)
            .join(totals, Corpus.id == totals.c.group_id)
            .filter(Corpus.is_active)
            .all()
        )  # type: ignore
