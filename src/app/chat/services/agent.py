from typing import List, Tuple

from fastapi import BackgroundTasks
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.app.chat.services.helpers import stringify_docs_content
from src.app.search.domain.models import EnhancedSearchQuery
from src.app.search.services.search import SearchService
from src.app.shared.domain.documents import EnhancedDocumentSlice
from src.app.shared.utils.decorators import log_time_and_error
from src.app.shared.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

NO_DOCS_INSTRUCTION = (
    "No documents were retrieved. Do not cite, number, link or mention any "
    "document or source in your answer (no [[Doc N]] markers, no titles, no "
    "URLs)."
)
NO_DOCUMENTS_FOUND = f"No relevant documents found. {NO_DOCS_INSTRUCTION}"
SEARCH_UNAVAILABLE = (
    "The search service is temporarily unavailable. Tell the user, in their "
    "language, that resources could not be retrieved right now. "
    f"{NO_DOCS_INSTRUCTION}"
)


async def _get_resources_about_sustainability(
    rag_query: str,
    config: RunnableConfig,
) -> Tuple[str, List[EnhancedDocumentSlice]]:
    """Core logic for getting relevant resources about sustainability from WeLearn database."""

    tool_called: list = config["configurable"].get("tool_called", [False])
    if tool_called[0]:
        return (
            "Search has already been performed. Use the documents already retrieved to answer the question.",
            [],
        )
    tool_called[0] = True

    qp = EnhancedSearchQuery(
        query=rag_query,
        sdg_filter=config["configurable"].get("sdg_filter"),
        corpora=config["configurable"].get("corpora"),
    )
    sp: SearchService = config["configurable"].get("sp")
    background_tasks: BackgroundTasks = config["configurable"].get("background_tasks")
    if not sp:
        logger.warning("No SearchService found.")
        return NO_DOCUMENTS_FOUND, []

    try:
        docs = await sp.search_handler(
            background_tasks=background_tasks, qp=qp, without_vectors=True
        )
    except Exception as e:
        # Fail soft so the agent can still answer instead of aborting the stream.
        logger.error("Search failed in agent tool: %s", e)
        return SEARCH_UNAVAILABLE, []
    if not docs:
        logger.warning("No documents found for the query.")
        return NO_DOCUMENTS_FOUND, []
    else:
        content = stringify_docs_content(docs[:7])  # Limit to first 7 documents
        return content, docs


@tool(response_format="content_and_artifact")
@log_time_and_error
async def get_resources_about_sustainability(
    rag_query: str,
    config: RunnableConfig,
) -> Tuple[str, List[EnhancedDocumentSlice]]:
    """Get relevant resources about sustainability from WeLearn database.

    Args:
        rag_query (str): The query string to search for relevant resources.
    """
    return await _get_resources_about_sustainability(rag_query, config)
