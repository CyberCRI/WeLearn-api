import numpy
from fastapi import APIRouter
from numpy import ndarray
from qdrant_client.http.models import models

from src.app.models.db_models import MetaDocument, MetaDocumentType
from src.app.models.documents import JourneySection, JourneySectionType
from src.app.services.search import SearchService
from src.app.services.sql_db import session_maker
from src.app.utils.logger import logger as logger_utils

router = APIRouter()
logger = logger_utils(__name__)

sp = SearchService()

def _flavored_with_subject(sdg_emb: ndarray, subject_emb: ndarray, discipline_factor: int = 2):
    embedding = sdg_emb + [
        discipline_factor * vec for vec in subject_emb.tolist()
    ]

    return embedding

@router.get(
    "/subject_list",
    summary="get subject's list",
    description="Retrieve all the subjects",
    response_model=list[str],
)
async def get_subject_list():
    ret = []
    with session_maker() as session:
        sdg_meta_documents: list[MetaDocument] = session.query(
            MetaDocument
        ).join(
            MetaDocumentType, MetaDocumentType.id == MetaDocument.meta_document_type_id
        ).filter(
            MetaDocumentType.title == "subject"
        ).all()

        ret = [md.title for md in sdg_meta_documents]

    return ret

@router.get(
    "/full_journey",
    summary="get the full journey",
    description="Get all documents for the micro learning journey of one sdg",
    # response_model=list[JourneySection],
)
async def get_full_journey(lang: str, sdg: int, subject: str):
    journey_part = [i.lower() for i in JourneySectionType]
    with session_maker() as session:
        sdg_meta_documents: list[MetaDocument] = session.query(
            MetaDocument
        ).join(
            MetaDocumentType, MetaDocumentType.id == MetaDocument.meta_document_type_id
        ).filter(
            MetaDocumentType.title.in_(journey_part), MetaDocument.sdg_related.contains([sdg])
        ).all()

        subject_meta_document: MetaDocument = session.query(
            MetaDocument
        ).join(
            MetaDocumentType, MetaDocumentType.id == MetaDocument.meta_document_type_id
        ).filter(
            MetaDocumentType.title == "subject", MetaDocument.title == subject
        ).first()

        if not subject_meta_document:
            raise ValueError(
                f"Subject '{subject}' not found in meta documents."
            )
        if not sdg_meta_documents:
            raise ValueError(
                f"SDG '{sdg}' not found in meta documents."
            )

        subject_binary_embedding = subject_meta_document.embedding
        if not isinstance(subject_binary_embedding, bytes):
            raise ValueError(
                f"Embedding must be of type bytes, received type: {type(subject_binary_embedding).__name__}"
            )
        subject_embedding: numpy.ndarray = numpy.frombuffer(
            bytes(subject_binary_embedding), dtype=numpy.float32
        )

        ret = {}
        for sdg_doc in sdg_meta_documents:
            sdg_binary_embedding = sdg_doc.embedding
            if not isinstance(sdg_binary_embedding, bytes):
                raise ValueError(
                    f"Embedding must be of type bytes, received type: {type(sdg_binary_embedding).__name__}"
                )

            sdg_embedding: numpy.ndarray = numpy.frombuffer(
                bytes(sdg_binary_embedding), dtype=numpy.float32
            )

            flavored_embedding = _flavored_with_subject(sdg_embedding, subject_embedding)

            sdg_filter = models.FieldCondition(
                            key="document_sdg",
                            match=models.MatchValue(value=sdg),
                        )

            try:
                sdg_doc_type = JourneySectionType[sdg_doc.meta_document_type.title.upper()]
            except KeyError:
                raise NotImplementedError(
                    f"Meta document type '{sdg_doc.meta_document_type.title}' is not a valid JourneySectionType."
                )

            if sdg_doc_type == JourneySectionType.INTRODUCTION:
                gte = 60.0
                lte = 100.0
            elif sdg_doc_type == JourneySectionType.TARGET:
                gte = 0.0
                lte = 60.0
            else:
                raise NotImplementedError(
                    f"Journey section type '{sdg_doc_type}' is not implemented."
                )

            if not sdg_doc.meta_document_type.title.lower() in ret:
                ret[sdg_doc.meta_document_type.title.lower()] = []


            qdrant_filter = models.Filter(
                    must=[
                        sdg_filter,
                        models.FieldCondition(
                            key="document_details.readability",
                            range=models.Range(
                                gt=None,
                                gte=gte,
                                lt=None,
                                lte=lte,
                            ),
                        )
                    ]
                )

            qdrant_return = await sp.search(
                collection_info="collection_welearn_en_all-minilm-l6-v2",
                embedding=flavored_embedding,
                filters=qdrant_filter,
                nb_results=10,
                with_vectors=False
            )

            if len(qdrant_return) > 0:
                ret[sdg_doc.meta_document_type.title.lower()].append({
                    "title": sdg_doc.title,
                    "content": sdg_doc.full_content,
                    "documents": qdrant_return
                })

    return ret