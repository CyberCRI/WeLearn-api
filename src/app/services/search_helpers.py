import asyncio
from typing import Awaitable, Callable, List

from fastapi import Response
from qdrant_client.http.models import ScoredPoint

from src.app.models.search import EnhancedSearchQuery
from src.app.services.exceptions import (
    handle_error,
)
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)



async def search_multi_inputs(
    response: Response,
    qp: EnhancedSearchQuery,
    callback_function: Callable[..., Awaitable[List[ScoredPoint]]],
):
    try:
        qps: list[EnhancedSearchQuery] = []
        for query in qp.query:
            temp_qp = qp.model_copy()
            temp_qp.query = query
            qps.append(temp_qp)



        tasks = [
            callback_function(
                qp=qp,
                method='by_slices',
            )
            for qp in qps
        ]

        all_data: list[ScoredPoint] = []
        for coroutine in asyncio.as_completed(tasks):
            data = await coroutine
            if data:
                all_data.extend(data)

        return all_data
    except Exception as e:
        handle_error(response=response, exc=e)
    return None


# todo: clean code
# andle tests
