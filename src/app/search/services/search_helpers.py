import asyncio
from typing import Awaitable, Callable

from fastapi import BackgroundTasks

from src.app.search.domain.models import (
    MAX_SEARCH_INPUTS,
    EnhancedSearchQuery,
    SearchMethods,
)
from src.app.shared.domain.documents import EnhancedDocumentSlice
from src.app.shared.domain.exceptions import TooManyQueriesError, handle_error
from src.app.shared.utils.logger import logger as logger_utils

logger = logger_utils(__name__)

# searches of one request running at the same time (embedding + Qdrant + DB)
MAX_CONCURRENT_SEARCHES = 4


async def search_multi_inputs(
    background_tasks: BackgroundTasks,
    qp: EnhancedSearchQuery,
    callback_function: Callable[..., Awaitable[list[EnhancedDocumentSlice]]],
) -> list[EnhancedDocumentSlice] | None:
    queries = [qp.query] if isinstance(qp.query, str) else qp.query
    if len(queries) > MAX_SEARCH_INPUTS:
        raise TooManyQueriesError(
            f"At most {MAX_SEARCH_INPUTS} queries can be searched at once"
        )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def bounded_search(
        single_qp: EnhancedSearchQuery,
    ) -> list[EnhancedDocumentSlice]:
        async with semaphore:
            return await callback_function(
                qp=single_qp,
                background_tasks=background_tasks,
                method=SearchMethods.BY_SLICES,
            )

    try:
        tasks = [
            bounded_search(qp.model_copy(update={"query": query})) for query in queries
        ]

        all_data: list[EnhancedDocumentSlice] = []
        for coroutine in asyncio.as_completed(tasks):
            data = await coroutine
            if data:
                all_data.extend(data)

        return all_data
    except Exception as e:
        handle_error(exc=e)
        return None
