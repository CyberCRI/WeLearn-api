from fastapi import APIRouter
from starlette.responses import PlainTextResponse

from src.app.bibliography.domain.models import DocumentIDs
from src.app.bibliography.services.ris import welearn_document_to_ris
from src.app.search.infra.queries import get_documents_by_ids

router = APIRouter()


@router.post(
    "/export_bibliography",
    response_class=PlainTextResponse,
    responses={200: {"content": {"text/plain": {"schema": {"type": "string"}}}}},
)
async def export_bibliography(body: DocumentIDs) -> str:
    documents_ids = [str(u) for u in body.documents_ids]
    docs = get_documents_by_ids(documents_ids)

    records = [welearn_document_to_ris(doc) for doc in docs]
    return "\n".join(records) + ("\n" if records else "")
