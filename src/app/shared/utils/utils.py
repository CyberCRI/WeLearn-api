import asyncio

from docx import Document as DocxReader
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pypdf import PdfReader

from src.app.shared.domain.documents import EnhancedDocumentSlice
from src.app.shared.domain.exceptions import (
    EmptyDocumentTextError,
    FileTooLargeError,
    TooManyFilesError,
)
from src.app.shared.infra.pdf_extractor import extract_txt_from_pdf_with_tika
from src.app.shared.utils.decorators import log_time_and_error
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as utils_logger

logger = utils_logger(__name__)
settings = get_settings()

# chunk size used to measure an upload whose size Starlette did not record
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024


def build_system_message(
    role: str,
    backstory: str,
    goal: str,
    instructions: str | None = None,
    expected_output: str | None = None,
) -> str:
    message = f"You are {role}. {backstory}\nYour personal goal is: {goal}."
    if instructions:
        message += (
            f"You must accomplish your goal by following these steps: {instructions}"
        )
    if expected_output:
        message += f"\nThis is the expected criteria for your final answer: {expected_output}\nYou MUST return the actual complete content as the final answer, not a summary."
    return message


def extract_doc_info(documents: list[EnhancedDocumentSlice]) -> list[dict]:
    """
    Extracts the document information from a list of documents.
    Args:
        documents (list[Document]): List of Document objects.
    Returns:
        list[dict]: List of dictionaries containing document information.
    """
    result = []
    for doc in documents:
        payload = getattr(doc, "payload", None)
        if payload is None and isinstance(doc, dict):
            payload = doc
        if payload is None:
            continue

        if isinstance(payload, dict):
            title = payload.get("document_title")
            url = payload.get("document_url")
            content = payload.get("slice_content")
        else:
            title = getattr(payload, "document_title", None)
            url = getattr(payload, "document_url", None)
            content = getattr(payload, "slice_content", None)

        if title is None and url is None and content is None:
            continue

        result.append(
            {
                "title": title,
                "url": url,
                "content": content,
            }
        )
    return result


async def _upload_size(file: UploadFile, limit: int) -> int:
    """
    Size of an upload in bytes. Starlette records it while parsing the form;
    otherwise read the file in chunks, stopping as soon as it passes `limit`.
    """
    if file.size is not None:
        return file.size

    size = 0
    await file.seek(0)
    while chunk := await file.read(UPLOAD_READ_CHUNK_SIZE):
        size += len(chunk)
        if size > limit:
            break
    await file.seek(0)
    return size


async def check_upload_limits(
    files: list[UploadFile], max_files: int, max_file_size: int
) -> None:
    """
    Reject an upload before any extraction: more than `max_files` files
    (TooManyFilesError, 400) or a file over `max_file_size` bytes
    (FileTooLargeError, 413).
    """
    if len(files) > max_files:
        raise TooManyFilesError(f"At most {max_files} files can be uploaded at once")

    for file in files:
        if await _upload_size(file, max_file_size) > max_file_size:
            raise FileTooLargeError(
                f"Each file must be at most {max_file_size // (1024 * 1024)} MB"
            )


@log_time_and_error
async def get_files_content(files: list[UploadFile]) -> list[str]:
    # gather keeps the upload order, so the joined text follows the files as sent
    files_content: list[str] = list(
        await asyncio.gather(*(get_file_content(file) for file in files))
    )

    if len(files_content) == 0:
        raise HTTPException(status_code=400, detail="added files are empty")

    return files_content


async def get_file_content(file: UploadFile) -> str:
    """Extract text content from various file types.

    Args:
        file (UploadFile): The uploaded file to process

    Returns:
        str: The extracted text content

    Raises:
        HTTPException: If file type is unsupported or content cannot be read
    """
    content_type = file.content_type

    content_extractors = {
        "application/pdf": _extract_pdf_content,
        "application/x-pdf": _extract_pdf_content,
        "text/plain": _extract_text_content,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _extract_docx_content,
    }
    if content_type not in content_extractors:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        content = await content_extractors[content_type](file)
    except Exception as e:
        # the parser's own message stays in the logs, not in the response
        logger.warning("file_unreadable content_type=%s error=%r", content_type, e)
        raise HTTPException(status_code=400, detail="Error reading file") from e
    # a scanned pdf has no text layer: a clear 422, not a read error
    if not content or not content.strip():
        raise EmptyDocumentTextError()
    return content.strip()


# the parsers are synchronous and slow on a large file: they run in the threadpool, so the
# event loop keeps serving every other request meanwhile


def _read_pdf(stream) -> str:
    reader = PdfReader(stream)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(stream) -> str:
    reader = DocxReader(stream)
    return "\n".join(paragraph.text for paragraph in reader.paragraphs)


async def _extract_pdf_content(file) -> str:
    try:
        return await extract_txt_from_pdf_with_tika(file.file, settings.TIKA_URL_BASE)
    except Exception as e:
        logger.warning("tika_failed falling back to pypdf error=%r", e)
        return await run_in_threadpool(_read_pdf, file.file)


async def _extract_text_content(file) -> str:
    content = await file.read()
    return content.decode("utf-8", errors="ignore")


async def _extract_docx_content(file) -> str:
    return await run_in_threadpool(_read_docx, file.file)
