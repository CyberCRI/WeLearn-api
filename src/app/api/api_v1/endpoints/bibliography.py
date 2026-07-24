from fastapi import APIRouter, BackgroundTasks, Request
from starlette.responses import PlainTextResponse

from src.app.models.bibliography import DocumentIDs
from src.app.services.helpers import welearn_document_to_ris
from src.app.services.sql_db.queries import get_documents_by_ids
from src.app.shared.utils.dependencies import get_settings
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()


@router.post("/export_bibliography", response_class=PlainTextResponse)
async def export_bibliography(
    request: Request,
    background_tasks: BackgroundTasks,
    body: DocumentIDs,
):
    documents_ids = [str(u) for u in body.documents_ids]

    docs = get_documents_by_ids(documents_ids)

    ret = ""
    for doc in docs:
        ret += welearn_document_to_ris(doc)
        ret += "\n"

    return ret
