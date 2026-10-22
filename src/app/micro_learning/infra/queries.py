# src/app/micro_learning/infra/queries.py
from uuid import UUID

from welearn_database.data.models import ContextDocument

from src.app.search.models.search import ContextType
from src.app.shared.domain.documents import JourneySection
from src.app.shared.infra.db import session_maker


def get_subjects(embedding_models_ids: list[UUID]) -> list[ContextDocument]:
    """
    Get all the subject meta documents from the database.
    Returns: List of subject meta documents.
    """
    with session_maker() as session:
        sdg_meta_documents: list[ContextDocument] = (
            session.query(ContextDocument)
            .filter(
                ContextDocument.context_type == ContextType.SUBJECT.value.lower(),
                ContextDocument.embedding_model_id.in_(embedding_models_ids),
            )
            .all()
        )
    return sdg_meta_documents


def get_context_documents(
    journey_part: list[JourneySection], sdg: int, embedding_models_ids: list[UUID]
):
    """
    Get the context documents from the database.

    Args:
        embedding_models_ids: Database IDs of embeddings models used for vectorize documents
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
                ContextDocument.embedding_model_id.in_(embedding_models_ids),
            )
            .all()
        )
    return sdg_meta_documents
