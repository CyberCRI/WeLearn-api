# src/app/services/sql_db/queries.py
import uuid
from collections import Counter
from threading import Lock
from uuid import UUID

from sqlalchemy import func, select
from welearn_database.data.enumeration import Step
from welearn_database.data.models import (
    ChatMessage,
    ContextDocument,
    Corpus,
    CorpusNameEmbeddingModelLang,
    DataCollectionCampaignManagement,
    DocumentSlice,
    EmbeddingModel,
    EndpointRequest,
    ErrorDataQuality,
    ProcessState,
    QtyDocumentInQdrant,
    QtyDocumentInQdrantPerCorpus,
    QtyDocumentPerCorpus,
    ReturnedDocument,
    Sdg,
    WeLearnDocument,
)

from src.app.models.chat import Role
from src.app.models.documents import Document, DocumentPayloadModel, JourneySection
from src.app.models.search import ContextType
from src.app.services.constants import APP_NAME
from src.app.services.sql_db.sql_service import session_maker

model_id_cache: dict[str, UUID] = {}
model_id_lock = Lock()


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
        )  # type: ignore


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


def register_endpoint(endpoint, session_id, http_code):
    with session_maker() as session:
        endpoint_request = EndpointRequest(
            endpoint_name=endpoint, session_id=session_id, http_code=http_code
        )
        session.add(endpoint_request)
        session.commit()


def get_subject(subject: str, embedding_model_id: UUID) -> ContextDocument | None:
    """
    Get the subject meta document from the database.
    Args:
        subject: The subject to get.

    Returns: The subject meta document.

    """
    with session_maker() as session:
        subject_meta_document: ContextDocument | None = (
            session.query(ContextDocument)
            .filter(
                ContextDocument.context_type == ContextType.SUBJECT.value.lower(),
                ContextDocument.title == subject,
                ContextDocument.embedding_model_id == embedding_model_id,
            )
            .first()
        )
    return subject_meta_document


def get_subjects(embedding_model_id: UUID) -> list[ContextDocument]:
    """
    Get all the subject meta documents from the database.
    Returns: List of subject meta documents.
    """
    with session_maker() as session:
        sdg_meta_documents: list[ContextDocument] = (
            session.query(ContextDocument)
            .filter(
                ContextDocument.context_type == ContextType.SUBJECT.value.lower(),
                ContextDocument.embedding_model_id == embedding_model_id,
            )
            .all()
        )
    return sdg_meta_documents


def get_context_documents(
    journey_part: JourneySection, sdg: int, embedding_model_id: UUID
):
    """
    Get the context documents from the database.

    Args:
        journey_part: The journey part to get the context documents for.
        sdg: The SDG to get the context documents for.
    Returns: List of context documents.
    """
    with session_maker() as session:
        sdg_meta_documents: list[ContextDocument] = (
            session.query(ContextDocument)
            .filter(
                ContextDocument.context_type.in_(journey_part),
                ContextDocument.sdg_related.contains([sdg]),
                ContextDocument.embedding_model_id == embedding_model_id,
            )
            .all()
        )
    return sdg_meta_documents


def get_embeddings_model_id_according_name(model_name: str) -> UUID | None:
    """
    Get the embeddings model ID according to its name.

    Args:
        model_name: The name of the embeddings model.

    Returns:
        The ID of the embeddings model if found, otherwise None.
    """
    with model_id_lock:
        if model_name in model_id_cache:
            return model_id_cache[model_name]

    with session_maker() as session:
        model = (
            session.query(EmbeddingModel)
            .filter(EmbeddingModel.title == model_name)
            .first()
        )
        return model.id if model else None


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


def write_user_query(
    user_id: UUID, query_content: str, conversation_id: UUID | None
) -> UUID:
    """
    Write a user query to the chat in the database.
    Args:
        user_id:  The ID of the user.
        query_content: The content of the user query.
        conversation_id: Key used for aggregated messages together, if None a new is generated in the API

    Returns:
        The ID of the new chat message entry.
    """
    if not conversation_id:
        conversation_id = uuid.uuid4()
    chat_msg = ChatMessage(
        inferred_user_id=user_id,
        role=Role.USER.value,
        textual_content=query_content,
        conversation_id=conversation_id,
    )
    with session_maker() as session:
        session.add(chat_msg)
        session.commit()
        return conversation_id


def write_chat_answer(
    user_id: UUID, answer: str, docs: list[Document], conversation_id: UUID
) -> UUID:
    """
    Write a chat answer to the database along with the referenced documents.
    Args:
        user_id: The ID of the user.
        answer: The content of the chat answer.
        docs: The list of documents referenced in the answer.
        conversation_id: Key used for aggregated messages together

    Returns:
        The ID of the new chat message entry and the list of returned document IDs.
    """
    chat_msg_id = uuid.uuid4()

    chat_msg = ChatMessage(
        id=chat_msg_id,
        inferred_user_id=user_id,
        role=Role.ASSISTANT.value,
        textual_content=answer,
        conversation_id=conversation_id,
    )
    returned_docs: list[ReturnedDocument] = []
    for doc in docs:
        returned_doc = ReturnedDocument(
            message_id=chat_msg_id,
            document_id=doc.payload.document_id,
        )
        returned_docs.append(returned_doc)

    with session_maker() as session:
        session.add(chat_msg)
        session.add_all(returned_docs)
        session.commit()
        return chat_msg.id


def update_returned_document_click(document_id: UUID, message_id: UUID) -> None:
    """
    Write a click on a returned document to the database.
    Args:
        document_id: The ID of the document that was clicked.
        message_id: The ID of the chat message associated with the document.

    Returns:
        None
    """
    with session_maker() as session:
        returned_doc = (
            session.query(ReturnedDocument)
            .filter(
                ReturnedDocument.document_id == document_id,
                ReturnedDocument.message_id == message_id,
            )
            .first()
        )
        if returned_doc:
            returned_doc.is_clicked = True
            session.commit()


def get_current_data_collection_campaign() -> DataCollectionCampaignManagement | None:
    with session_maker() as session:
        campaign = (
            session.query(DataCollectionCampaignManagement)
            .filter(DataCollectionCampaignManagement.end_at > func.now())
            .order_by(DataCollectionCampaignManagement.end_at)
            .first()
        )
        return campaign
