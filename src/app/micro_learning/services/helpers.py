from fastapi import HTTPException
from qdrant_client.http.models import models
from welearn_database.data.models import EmbeddingModel

from src.app.shared.domain.collections import Collection
from src.app.shared.domain.documents import JourneySectionType
from src.app.shared.infra.queries import get_embeddings_model_id_according_name
from src.app.shared.utils.decorators import log_time_and_error_sync


def choose_readability_according_journey_section_type(
    sdg_doc_type: JourneySectionType,
) -> models.Range:
    """
    Choose the readability range according to the journey section type.
    1. Introduction: 60-100 (easier)
    2. Target: 0-60 (harder)
    Args:
        sdg_doc_type: The journey section type.

    Returns: The readability range for Qdrant search.

    """
    if sdg_doc_type == JourneySectionType.INTRODUCTION:
        readability_range = models.Range(
            gte=60,
            lte=100,
        )
    elif sdg_doc_type == JourneySectionType.TARGET:
        readability_range = models.Range(
            gte=0,
            lte=60,
        )
    else:
        raise NotImplementedError(
            f"Journey section type '{sdg_doc_type}' is not implemented."
        )
    return readability_range


@log_time_and_error_sync
async def collection_and_model_id_according_lang(
    lang: str | None, sp
) -> tuple[Collection, list[EmbeddingModel]]:
    """
    Get the collection info and model id according to the language.
    Args:
        sp: The search service.
        lang: The language to get the collection info and model id for. If None, the default language is used wich is multilingual.

    Returns:
        A tuple of the collection info and model id.
    """
    if lang:
        collection_info = await sp.get_collection_by_language(lang)
    else:
        collection_info = await sp.get_collection_by_language()

    if not collection_info:
        raise HTTPException(
            status_code=404, detail=f"No collection found for language '{lang}'."
        )
    emb_models = get_embeddings_model_id_according_name(collection_info.model)
    if not emb_models:
        raise ValueError(
            f"Embedding model '{collection_info.model}' not found in the database."
        )
    return collection_info, emb_models
