from fastapi import APIRouter
from starlette.responses import PlainTextResponse

from src.app.bibliography.models.bibliography import DocumentIDs
from src.app.bibliography.helpers.helpers import welearn_document_to_ris
from src.app.services.sql_db.queries import get_documents_by_ids

router = APIRouter()


@router.post("/export_bibliography", response_class=PlainTextResponse)
async def export_bibliography(body: DocumentIDs) -> str:
    documents_ids = [str(u) for u in body.documents_ids]
    docs = get_documents_by_ids(documents_ids)

    records = [welearn_document_to_ris(doc) for doc in docs]
    return "\n".join(records) + ("\n" if records else "")
