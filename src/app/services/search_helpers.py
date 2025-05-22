import asyncio
from typing import Awaitable, Callable

from qdrant_client.http.models import ScoredPoint

from src.app.models.search import EnhancedSearchQuery, SearchMethods
from src.app.services.exceptions import handle_error
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


async def search_multi_inputs(
    qp: EnhancedSearchQuery,
    callback_function: Callable[..., Awaitable[list[ScoredPoint]]],
) -> list[ScoredPoint] | None:
    try:
        qps: list[EnhancedSearchQuery] = []
        for query in qp.query:
            temp_qp = qp.model_copy()
            temp_qp.query = query
            qps.append(temp_qp)

        tasks = [
            callback_function(
                qp=qp,
                method=SearchMethods.BY_SLICES,
            )
            for qp in qps
        ]

        data = await asyncio.gather(*tasks)
        all_data: list[ScoredPoint] = []

        for sublist in data:
            all_data.extend(sublist)

        return all_data
    except Exception as e:
        handle_error(exc=e)
        return None
