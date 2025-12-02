import asyncio

from docx import Document as DocxReader
from fastapi import HTTPException, UploadFile

# from src.app.services.pdf_extractor import extract_txt_from_pdf_with_tika
from pypdf import PdfReader
from qdrant_client.models import ScoredPoint

from src.app.api.dependencies import get_settings
from src.app.utils.decorators import log_time_and_error_sync

settings = get_settings()


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


def extract_doc_info(documents: list[ScoredPoint]) -> list[dict]:
    """
    Extracts the document information from a list of documents.
    Args:
        documents (list[Document]): List of Document objects.
    Returns:
        list[dict]: List of dictionaries containing document information.
    """
    return [
        {
            "title": getattr(doc.payload, "document_title", ""),  # type: ignore
            "url": getattr(doc.payload, "document_url", ""),  # type: ignore
            "content": getattr(doc.payload, "slice_content", ""),  # type: ignore
        }
        for doc in documents
        if doc.payload is not None
    ]


@log_time_and_error_sync
async def get_files_content(files: list[UploadFile]) -> list[str]:
    files_content: list[str] = []
    tasks = [get_file_content(file) for file in files]

    for coroutine in asyncio.as_completed(tasks):
        content = await coroutine
        files_content.append(content)

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
        if not content:
            raise HTTPException(status_code=400, detail="Unable to extract content")
        return content.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")


async def _extract_pdf_content(file) -> str:
    reader = PdfReader(file.file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
    # content = extract_txt_from_pdf_with_tika(file.file, settings.TIKA_URL_BASE)

    # return content


async def _extract_text_content(file) -> str:
    content = await file.read()
    return content.decode("utf-8", errors="ignore")


async def _extract_docx_content(file) -> str:
    reader = DocxReader(file.file)
    return "\n".join(paragraph.text for paragraph in reader.paragraphs)
