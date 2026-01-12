# src/app/services/sql_db/queries.py

from collections import Counter

from sqlalchemy import select
from welearn_database.data.models import (
    Corpus,
    CorpusNameEmbeddingModelLang,
    DocumentSlice,
    QtyDocumentInQdrant,
    QtyDocumentInQdrantPerCorpus,
    QtyDocumentPerCorpus,
    Sdg,
    WeLearnDocument,
)

from src.app.models.documents import Document, DocumentPayloadModel
from src.app.services.sql_service import session_maker


def get_collections_sync():
    statement = select(
        CorpusNameEmbeddingModelLang.source_name,
        CorpusNameEmbeddingModelLang.lang,
        CorpusNameEmbeddingModelLang.title,
    )
    with session_maker() as s:
        return s.execute(statement).all()


def get_nb_docs_sync():
    statement = select(QtyDocumentInQdrant.document_in_qdrant)
    with session_maker() as s:
        return s.execute(statement).first()


def get_document_qty_table_info_sync() -> (
    list[tuple[Corpus, QtyDocumentInQdrantPerCorpus, QtyDocumentPerCorpus]]
):
    with session_maker() as s:
        return (
            s.query(Corpus, QtyDocumentInQdrantPerCorpus, QtyDocumentPerCorpus)
            .join(
                QtyDocumentInQdrantPerCorpus,
                Corpus.source_name == QtyDocumentInQdrantPerCorpus.source_name,
            )
            .join(
                QtyDocumentPerCorpus,
                Corpus.source_name == QtyDocumentPerCorpus.source_name,
            )
            .all()
        )


def get_documents_payload_by_ids_sync(documents_ids: list[str]) -> list[Document]:
    with session_maker() as s:
        documents = s.execute(
            select(
                WeLearnDocument.title,
                WeLearnDocument.url,
                WeLearnDocument.corpus_id,
                WeLearnDocument.id,
                WeLearnDocument.description,
                WeLearnDocument.details,
            ).where(WeLearnDocument.id.in_(documents_ids))
        ).all()

        # Batch fetch corpora
        corpus_ids = list({doc.corpus_id for doc in documents})
        corpora = s.execute(
            select(Corpus.id, Corpus.source_name).where(Corpus.id.in_(corpus_ids))
        ).all()
        corpus_map = {corpus.id: corpus.source_name for corpus in corpora}

        # Batch fetch slices
        slices = s.execute(
            select(DocumentSlice.id, DocumentSlice.document_id).where(
                DocumentSlice.document_id.in_(documents_ids)
            )
        ).all()
        slices_ids_map = {}
        slice_ids = []
        for slice_ in slices:
            slices_ids_map.setdefault(slice_.document_id, []).append(slice_.id)
            slice_ids.append(slice_.id)

        # Batch fetch SDGs
        sdgs = s.execute(
            select(Sdg.sdg_number, Sdg.slice_id).where(Sdg.slice_id.in_(slice_ids))
        ).all()
        sdg_map = {}
        for sdg in sdgs:
            sdg_map.setdefault(sdg.slice_id, []).append(sdg)

        # Compose documents
        docs = []
        for doc in documents:
            corpus = corpus_map.get(doc.corpus_id)
            slices_id_for_doc = slices_ids_map.get(doc.id, [])
            sdgs_for_doc = []
            for slice_id in slices_id_for_doc:
                sdgs_for_doc.extend(sdg_map.get(slice_id, []))
            short_sdg_list = Counter(
                [sdg.sdg_number for sdg in sdgs_for_doc]
            ).most_common(2)
            docs.append(
                Document(
                    score=0.0,
                    payload=DocumentPayloadModel(
                        document_id=doc.id,
                        document_title=doc.title,
                        document_url=doc.url,
                        document_desc=doc.description,
                        document_sdg=[sdg[0] for sdg in short_sdg_list],
                        document_details=doc.details,
                        slice_content="",
                        document_lang="",
                        document_corpus=corpus if corpus else "",
                        slice_sdg=None,
                    ),
                )
            )
        return docs
