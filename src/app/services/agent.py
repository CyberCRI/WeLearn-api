from typing import Tuple, List

from langchain_core.tools import tool

from src.app.models.documents import Document
from src.app.models.search import EnhancedSearchQuery
from src.app.services.helpers import stringify_docs_content
from src.app.services.search import SearchService
from src.app.utils.decorators import log_time_and_error
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


@tool(response_format="content_and_artifact")
@log_time_and_error
async def get_resources_about_sustainability(
    rag_query: str,
) -> Tuple[str, List[Document]]:
    """Get relevant resources about sustainability from WeLearn database.

    Args:
        rag_query (str): The query string to search for relevant resources.
    """
    sp = SearchService()
    qp = EnhancedSearchQuery(
        query=rag_query,
        sdg_filter=None,        
    )
    docs = await sp.search_handler(qp)
    content = stringify_docs_content(docs[:7])  # Limit to first 7 documents
    return content, docs
