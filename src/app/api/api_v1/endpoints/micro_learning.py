import numpy
from fastapi import APIRouter, HTTPException

from welearn_database.data.models import ContextDocument
from src.app.models.documents import JourneySectionType
from src.app.models.search import SearchFilters
from src.app.services.helpers import (
    choose_readability_according_journey_section_type,
    collection_and_model_id_according_lang,
    convert_embedding_bytes,
)
from src.app.services.search import SearchService
from src.app.services.sql_db import get_context_documents, get_subject, get_subjects
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)

sp = SearchService()


@router.get(
    "/subject_list",
    summary="get subject's list",
    description="Retrieve all the subjects",
    response_model=list[str],
)
async def get_subject_list(lang: str | None = None) -> list[str]:
    collection_info, model_id = await collection_and_model_id_according_lang(
        sp=sp, lang=lang
    )
    ret = [md.title for md in get_subjects(embedding_model_id=model_id)]
    if len(ret) == 0:
        raise HTTPException(status_code=404, detail="No subjects found.")
    return ret


@router.get(
    "/full_journey",
    summary="get the full journey",
    description="Get all documents for the micro learning journey of one sdg",
)
async def get_full_journey(sdg: int, subject: str, lang: str | None = None):
    collection_info, model_id = await collection_and_model_id_according_lang(
        sp=sp, lang=lang
    )

    journey_part = [i.lower() for i in JourneySectionType]
    sdg_meta_documents = get_context_documents(
        journey_part=journey_part, sdg=sdg, embedding_model_id=model_id
    )
    if not sdg_meta_documents:
        raise HTTPException(
            status_code=404, detail=f"SDG '{sdg}' not found in meta documents."
        )

    subject_meta_document: ContextDocument | None = get_subject(
        subject=subject, embedding_model_id=model_id
    )

    if not subject_meta_document:
        raise ValueError(f"Subject '{subject}' not found in meta documents.")

    subject_binary_embedding = subject_meta_document.embedding
    if not isinstance(subject_binary_embedding, bytes):
        raise ValueError(
            f"Embedding must be of type bytes, received type: {type(subject_binary_embedding).__name__}"
        )
    subject_embedding = convert_embedding_bytes(
        embeddings_byte=subject_binary_embedding
    )

    ret = {}
    for sdg_doc in sdg_meta_documents:
        sdg_binary_embedding = sdg_doc.embedding
        if not isinstance(sdg_binary_embedding, bytes):
            raise ValueError(
                f"Embedding must be of type bytes, received type: {type(sdg_binary_embedding).__name__}"
            )

        sdg_embedding: numpy.ndarray = convert_embedding_bytes(
            embeddings_byte=sdg_binary_embedding
        )

        flavored_embedding = sp.flavored_with_subject(sdg_embedding, subject_embedding)

        try:
            sdg_doc_type = JourneySectionType[sdg_doc.context_type.upper()]
        except KeyError:
            raise NotImplementedError(
                f"Meta document type '{sdg_doc.context_type}' is not a valid JourneySectionType."
            )

        readability_range = choose_readability_according_journey_section_type(
            sdg_doc_type
        )

        if not sdg_doc.context_type.lower() in ret:
            ret[sdg_doc.context_type.lower()] = []

        qdrant_filter = SearchFilters(
            slice_sdg=[sdg], readability=readability_range, document_corpus=None
        ).build_filters()

        qdrant_return = await sp.search(
            collection_info=collection_info.name,
            embedding=flavored_embedding,
            filters=qdrant_filter,
            nb_results=10,
            with_vectors=False,
        )

        if len(qdrant_return) > 0:
            ret[sdg_doc.context_type.lower()].append(
                {
                    "title": sdg_doc.title,
                    "content": sdg_doc.full_content,
                    "documents": qdrant_return,
                }
            )

    return ret
