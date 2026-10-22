# src/app/metric/infra/queries.py
from sqlalchemy import func
from welearn_database.data.models import (
    Corpus,
    QtyDocumentInQdrantPerCorpus,
    QtyDocumentPerCorpus,
)

from src.app.shared.infra.db import session_maker


def get_document_qty_table_info_sync() -> list[tuple[Corpus, int, int]]:
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
