import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from src.app.search.domain.models import (
    MAX_SEARCH_INPUTS,
    EnhancedSearchQuery,
    SearchMethods,
)
from src.app.search.services import search_helpers
from src.app.search.services.search_helpers import search_multi_inputs
from src.app.shared.domain.exceptions import (
    CollectionNotFoundError,
    TooManyQueriesError,
)


class SearchMultiInputsTests(IsolatedAsyncioTestCase):
    async def test_searches_each_query(self):
        callback = AsyncMock(side_effect=lambda qp, **_: [qp.query])
        qp = EnhancedSearchQuery(query=["a", "b", "c"], nb_results=5)

        result = await search_multi_inputs(MagicMock(), qp, callback)

        self.assertCountEqual(result, ["a", "b", "c"])
        self.assertEqual(callback.await_count, 3)
        for call in callback.await_args_list:
            self.assertEqual(call.kwargs["method"], SearchMethods.BY_SLICES)
            self.assertEqual(call.kwargs["qp"].nb_results, 5)
        # the caller's query is left as it was
        self.assertEqual(qp.query, ["a", "b", "c"])

    async def test_single_string_query_is_one_search(self):
        callback = AsyncMock(return_value=["doc"])
        qp = EnhancedSearchQuery(query="one query")

        result = await search_multi_inputs(MagicMock(), qp, callback)

        self.assertEqual(result, ["doc"])
        callback.assert_awaited_once()
        self.assertEqual(callback.await_args.kwargs["qp"].query, "one query")

    async def test_too_many_queries_is_rejected_before_searching(self):
        callback = AsyncMock(return_value=[])
        qp = EnhancedSearchQuery(query=["q"] * (MAX_SEARCH_INPUTS + 1))

        with self.assertRaises(TooManyQueriesError) as ctx:
            await search_multi_inputs(MagicMock(), qp, callback)

        self.assertEqual(ctx.exception.status_code, 400)
        callback.assert_not_awaited()

    async def test_max_queries_is_accepted(self):
        callback = AsyncMock(return_value=[])
        qp = EnhancedSearchQuery(query=["q"] * MAX_SEARCH_INPUTS)

        await search_multi_inputs(MagicMock(), qp, callback)

        self.assertEqual(callback.await_count, MAX_SEARCH_INPUTS)

    async def test_semaphore_caps_concurrent_searches(self):
        active = 0
        peak = 0

        async def slow_search(**_):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return []

        qp = EnhancedSearchQuery(query=["q"] * MAX_SEARCH_INPUTS)

        await search_multi_inputs(MagicMock(), qp, slow_search)

        self.assertEqual(peak, search_helpers.MAX_CONCURRENT_SEARCHES)

    async def test_domain_error_becomes_http_exception(self):
        callback = AsyncMock(side_effect=CollectionNotFoundError())
        qp = EnhancedSearchQuery(query=["q"])

        with self.assertRaises(HTTPException) as ctx:
            await search_multi_inputs(MagicMock(), qp, callback)

        self.assertEqual(ctx.exception.status_code, 404)
