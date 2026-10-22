# src/app/search/infra/queries.py
import uuid
from collections import Counter
from uuid import UUID

from sqlalchemy import Row, and_, select
from sqlalchemy.orm import Session, joinedload
from welearn_database.data.enumeration import Step
from welearn_database.data.models import (
    Category,
    ContextDocument,
    Corpus,
    DocumentSlice,
    ErrorDataQuality,
    ProcessState,
    QtyDocumentInQdrant,
    Sdg,
    WeLearnDocument,
)

from src.app.search.domain.models import ContextType
from src.app.shared.domain.collections import CorpusRelation
from src.app.shared.domain.constants import APP_NAME
from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)
from src.app.shared.infra.db import session_maker

main_corpus_dict: dict[UUID, Corpus | None] = {}


def get_collections_info_sync():
    # Get active parent corpus
    stmt = (
        select(
            Corpus.source_name, Corpus.is_active, Category.title.label("category_name")
        )
        .outerjoin(Category, Corpus.category_id == Category.id)
        .where(and_(Corpus.is_active, Corpus.parent_corpus_id.is_(None)))
    )

    with session_maker() as session:
        results = session.execute(stmt).all()
        return results


def get_nb_docs_sync():
    statement = select(QtyDocumentInQdrant.document_in_qdrant)
    with session_maker() as s:
        return s.execute(statement).first()


def get_documents_by_ids(documents_ids: list[str]) -> list[WeLearnDocument]:
    with session_maker() as s:
        documents = (
            s.query(WeLearnDocument)
            .where(WeLearnDocument.id.in_(documents_ids))
            .options(joinedload(WeLearnDocument.corpus))
            .all()
        )

    ret = list(documents)

    return ret


def get_corpus_and_sub_corpus_repartition(corpus: Corpus) -> CorpusRelation:
    with session_maker() as s:

        if corpus.id not in main_corpus_dict:
            req_corpus: Corpus | None = (
                s.query(Corpus).filter(Corpus.id == corpus.parent_corpus_id).first()
            )
            main_corpus_dict[corpus.id] = req_corpus
        main_corpus = main_corpus_dict[corpus.id]

        if main_corpus:
            return CorpusRelation(sub_corpus=corpus, corpus=main_corpus)

        return CorpusRelation(
            corpus=corpus,
        )


def get_documents_according_slice_ids(
    slice_ids: list[str],
) -> dict[WeLearnDocument, list[DocumentSlice]]:
    ret: dict[WeLearnDocument, list[DocumentSlice]] = {}
    stmt = (
        select(DocumentSlice, WeLearnDocument)
        .join(WeLearnDocument, WeLearnDocument.id == DocumentSlice.document_id)
        .where(DocumentSlice.id.in_(slice_ids))
        .options(joinedload(WeLearnDocument.corpus))
    )
    with session_maker() as s:
        db_ret = s.execute(stmt).all()
        for wl_slice, wl_document in db_ret:
            if wl_document not in ret:
                ret[wl_document] = []
            ret[wl_document].append(wl_slice)
    return ret


def get_documents_payload_by_ids_sync(
    documents_ids: list[str],
) -> list[EnhancedDocumentSlice]:
    with session_maker() as s:
        documents: list[WeLearnDocument] = get_documents_by_ids(
            documents_ids=documents_ids
        )

        # Batch fetch slices
        slice_ids, slices_ids_map = batch_fetch_slices(documents_ids, s)

        # Batch fetch SDGs
        sdg_map = batch_fetch_sdgs(slice_ids)

        # Compose documents
        docs = []
        for doc in documents:
            corpus_relation = get_corpus_and_sub_corpus_repartition(corpus=doc.corpus)
            slices_id_for_doc = slices_ids_map.get(doc.id, [])
            sdgs_for_doc = []
            for slice_id in slices_id_for_doc:
                sdgs_for_doc.extend(sdg_map.get(slice_id, []))
            short_sdg_list = Counter(
                [sdg.sdg_number for sdg in sdgs_for_doc]
            ).most_common(2)
            docs.append(
                EnhancedDocumentSlice(
                    score=0.0,
                    payload=EnhancedDocumentSlicePayloadModel(
                        document_id=doc.id,
                        document_title=doc.title,
                        document_url=doc.url,
                        document_desc=doc.description,
                        document_external_id=doc.external_id,
                        document_sdg=[sdg[0] for sdg in short_sdg_list],
                        document_details=doc.details,
                        document_lang=doc.lang,
                        document_corpus=corpus_relation.corpus.source_name,
                        document_sub_corpus=(
                            corpus_relation.sub_corpus.source_name
                            if corpus_relation.sub_corpus
                            else None
                        ),
                        slice_sdg=None,
                        slice_content=None,
                    ),
                )
            )
        return docs


def batch_fetch_sdgs(slice_ids: list[UUID]) -> dict[uuid.UUID, list[Sdg]]:
    with session_maker() as s:
        db_ret_sdgs: tuple[Row[Sdg]] = s.execute(
            select(Sdg).where(Sdg.slice_id.in_(slice_ids))
        ).all()
        sdgs = [s[0] for s in db_ret_sdgs]
        sdg_map = {}
        for sdg in sdgs:
            sdg_map.setdefault(sdg.slice_id, []).append(sdg)
        return sdg_map


def batch_fetch_slices(
    documents_ids: list[str], s: Session
) -> tuple[list[UUID], dict[UUID, list[UUID]]]:
    """
    This method create a dict with the form <document_id: slice id list> and a list of slice ids
    :param documents_ids: list of document ids
    :param s: SQLAlchemy db session
    :return: tuple[list[UUID], dict[UUID, list[UUID]]]
    """
    slices = s.execute(
        select(DocumentSlice.id, DocumentSlice.document_id).where(
            DocumentSlice.document_id.in_(documents_ids)
        )
    ).all()

    slices_ids_map: dict[UUID, list[UUID]] = {}
    slice_ids: list[UUID] = []
    for slice_ in slices:
        slices_ids_map.setdefault(slice_.document_id, []).append(slice_.id)
        slice_ids.append(slice_.id)
    return slice_ids, slices_ids_map


def get_subject(
    subject: str, embedding_models_ids: list[UUID]
) -> ContextDocument | None:
    """
    Get the subject meta document from the database.
    Args:
        embedding_models_ids: Database IDs of embeddings models used for vectorize documents
        subject: The subject to get.

    Returns: The subject meta document.

    """
    with session_maker() as session:
        subject_meta_document: ContextDocument | None = (
            session.query(ContextDocument)
            .filter(
                ContextDocument.context_type == ContextType.SUBJECT.value.lower(),
                ContextDocument.title == subject,
                ContextDocument.embedding_model_id.in_(embedding_models_ids),
            )
            .first()
        )
    return subject_meta_document


def write_new_data_quality_error(
    document_id: UUID, error_info: str, slice_id: UUID | None = None
) -> UUID:
    """
    Write a new data quality error to the database.
    Args:
        document_id: The ID of the document with the error.
        slice_id: The ID of the document slice with the error.
        error_info:  The error information. Usually exception message.

    Returns:
        The ID of the new error entry.
    """
    with session_maker() as session:
        error_entry = ErrorDataQuality(
            id=uuid.uuid4(),
            document_id=document_id,
            slice_id=slice_id,
            error_raiser=APP_NAME,
            error_info=error_info,
        )
        session.add(error_entry)
        session.commit()
        return error_entry.id


def write_process_state(document_id: UUID, process_state: Step) -> UUID:
    """
    Write the process state of a document to the database.
    Args:
        document_id: The ID of the document.
        process_state: The current process state.

    Returns:
        The ID of the new process state entry.
    """
    with session_maker() as session:
        process_state_entry = ProcessState(
            id=uuid.uuid4(),
            document_id=document_id,
            title=process_state.value.lower(),
        )
        session.add(process_state_entry)
        session.commit()
        return process_state_entry.id
