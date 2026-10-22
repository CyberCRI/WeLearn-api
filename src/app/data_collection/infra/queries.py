# src/app/data_collection/infra/queries.py
import uuid
from uuid import UUID

from qdrant_client.http.models import ScoredPoint
from sqlalchemy import func
from welearn_database.data.models import (
    AnalyticForm,
    ChatMessage,
    DataCollectionCampaignManagement,
    FilterType,
    FilterUsedInQuery,
    QuestionType,
    ReturnedDocument,
)

from src.app.models.chat import Role
from src.app.shared.domain.documents import (
    EnhancedDocumentSlice,
    EnhancedDocumentSlicePayloadModel,
)
from src.app.shared.infra.db import session_maker


def register_form_answers_sync(
    form_name: str,
    question: str,
    answer: str,
    answer_type: QuestionType,
    session_id: UUID,
):
    with session_maker() as session:
        form_ans = AnalyticForm(
            form_name=form_name,
            question=question,
            answer=answer,
            answer_type=answer_type,
            session_id=session_id,
        )
        session.add(form_ans)
        session.commit()


def write_user_query(
    user_id: UUID,
    query_content: str,
    conversation_id: UUID | None,
    feature: str | None = None,
) -> tuple[UUID, UUID]:
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
        original_feature_name=feature,
    )
    with session_maker() as session:
        session.add(chat_msg)
        session.commit()
        return conversation_id, chat_msg.id


def write_chat_answer(
    user_id: UUID,
    answer: str,
    docs: list[EnhancedDocumentSlice | ScoredPoint] | None,
    conversation_id: UUID,
    feature: str | None = None,
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
        original_feature_name=feature,
    )

    with session_maker() as session:
        session.add(chat_msg)
        session.commit()

    if docs:
        write_returned_docs(chat_msg_id, docs)

    return chat_msg_id


def get_last_syllabus_conversation_id(user_id: UUID) -> UUID | None:
    """
    Get the conversation ID of the last syllabus creation message of the user.

    Args:
        user_id: The ID of the user.

    Returns:
        The conversation ID if found, otherwise None.
    """
    with session_maker() as session:
        last_message = (
            session.query(ChatMessage)
            .filter(
                ChatMessage.inferred_user_id == user_id,
                ChatMessage.original_feature_name == "syllabus_creation",
            )
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        return last_message.conversation_id if last_message else None


def get_last_syllabus_id_for_user(user_id: UUID) -> UUID | None:
    with session_maker() as session:
        last_message = (
            session.query(ChatMessage)
            .filter(
                ChatMessage.inferred_user_id == user_id,
                ChatMessage.original_feature_name.in_(
                    ["syllabus_creation", "syllabus_feedback", "syllabus_user_update"]
                ),
            )
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        return last_message.id if last_message else None


def update_syllabus_retrieved_status(syllabus_id: UUID) -> None:
    with session_maker() as session:
        syllabus_message = (
            session.query(ChatMessage).filter(ChatMessage.id == syllabus_id).first()
        )
        if syllabus_message:
            syllabus_message.is_retrieved_by_user = True
            session.commit()


def write_returned_docs(
    message_id: UUID,
    docs: list[EnhancedDocumentSlice | ScoredPoint],
    is_clicked: bool = False,
) -> None:
    """
    Register the returned documents for a chat message in the database.
    Args:
        message_id: The ID of the chat message.
        docs: The list of documents to register.
    """
    returned_docs = []
    for doc in docs:
        if isinstance(doc, ScoredPoint):
            doc = EnhancedDocumentSlice(
                score=0.0,
                payload=EnhancedDocumentSlicePayloadModel(**doc.payload),
            )

        returned_doc = ReturnedDocument(
            message_id=message_id,
            document_id=doc.payload.document_id,
            is_clicked=is_clicked,
        )
        returned_docs.append(returned_doc)

    with session_maker() as session:
        session.add_all(returned_docs)
        session.commit()


def write_filters_search(
    message_id: UUID, sdg_filter: list[int] | None, corpora: list[str] | None
) -> None:
    filter_entries = []
    for sdg in sdg_filter or []:
        filter_entry = FilterUsedInQuery(
            message_id=message_id,
            filter_type=FilterType.SDG.value,
            filter_value=str(sdg),
        )
        filter_entries.append(filter_entry)
    for corpus in corpora or []:
        filter_entry = FilterUsedInQuery(
            message_id=message_id,
            filter_type=FilterType.SOURCE.value,
            filter_value=corpus,
        )
        filter_entries.append(filter_entry)

    with session_maker() as session:
        session.add_all(filter_entries)
        session.commit()


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
