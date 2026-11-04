import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from baml_py.errors import (
    BamlClientFinishReasonError,
    BamlClientHttpError,
    BamlTimeoutError,
    BamlValidationError,
)

from src.app.tutor.services.b_agents.base import AgentError, BaseAgent

SLEEP_PATH = "src.app.tutor.services.b_agents.base.asyncio.sleep"


def validation_error() -> BamlValidationError:
    return BamlValidationError("prompt", "Failed to parse LLM response", "{", "")


def finish_reason_error() -> BamlClientFinishReasonError:
    return BamlClientFinishReasonError("prompt", "cut", "{", "length", "")


def http_error(status_code: int) -> BamlClientHttpError:
    return BamlClientHttpError("MistralSmall", "Request failed", status_code, "")


class TestBaseAgentRetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = BaseAgent(name="test_agent", max_retries=3)
        sleep_patcher = patch(SLEEP_PATH, new_callable=AsyncMock)
        self.mock_sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def test_default_max_retries_is_small(self):
        self.assertEqual(BaseAgent(name="default").max_retries, 2)

    async def test_success_first_try_awaits_and_returns_result(self):
        func = AsyncMock(return_value="ok")

        result = await self.agent.with_retry(func, "a", key="b")

        self.assertEqual(result, "ok")
        func.assert_awaited_once_with("a", key="b")
        self.mock_sleep.assert_not_awaited()

    async def test_validation_error_then_success_retries_and_returns(self):
        func = AsyncMock(side_effect=[validation_error(), finish_reason_error(), "ok"])

        result = await self.agent.with_retry(func)

        self.assertEqual(result, "ok")
        self.assertEqual(func.await_count, 3)
        self.assertEqual([c.args[0] for c in self.mock_sleep.await_args_list], [1, 2])

    async def test_persistent_validation_error_raises_agent_error(self):
        errors = [validation_error() for _ in range(3)]
        func = AsyncMock(side_effect=errors)

        with self.assertRaises(AgentError) as ctx:
            await self.agent.with_retry(func)

        self.assertEqual(func.await_count, 3)
        # no sleep after the last attempt
        self.assertEqual(self.mock_sleep.await_count, 2)
        self.assertIs(ctx.exception.__cause__, errors[-1])
        self.assertIn(
            "test_agent failed after 3 attempts: BamlValidationError",
            str(ctx.exception),
        )

    async def test_single_attempt_never_sleeps(self):
        agent = BaseAgent(name="once", max_retries=1)
        func = AsyncMock(side_effect=validation_error())

        with self.assertRaises(AgentError):
            await agent.with_retry(func)

        func.assert_awaited_once()
        self.mock_sleep.assert_not_awaited()

    async def test_non_retryable_errors_propagate_immediately_unchanged(self):
        # Transport errors are retried by the BAML client retry_policy, client
        # errors and programming errors are not worth retrying: none of them is
        # retried here, wrapped, or delayed.
        cases = [
            http_error(400),
            http_error(401),
            http_error(429),
            http_error(503),
            BamlTimeoutError("MistralSmall", "Request timed out"),
            ValueError("bug"),
            KeyError("missing"),
        ]
        for error in cases:
            with self.subTest(error=repr(error)):
                func = AsyncMock(side_effect=error)

                with self.assertRaises(type(error)) as ctx:
                    await self.agent.with_retry(func)

                self.assertIs(ctx.exception, error)
                func.assert_awaited_once()
                self.mock_sleep.assert_not_awaited()

    async def test_does_not_block_the_event_loop_with_time_sleep(self):
        func = AsyncMock(side_effect=[validation_error(), "ok"])

        with patch("src.app.tutor.services.b_agents.base.time.sleep") as time_sleep:
            await self.agent.with_retry(func)

        time_sleep.assert_not_called()
        self.mock_sleep.assert_awaited_once_with(1)


class TestLogCall(unittest.TestCase):
    def setUp(self):
        self.agent = BaseAgent(name="logger_agent")
        self.agent.logger = MagicMock()

    def test_log_call_without_tokens(self):
        self.agent.log_call({"text": "a" * 200}, [1, 2], 1.5)

        message = self.agent.logger.info.call_args.args[0]
        self.assertIn("'agent': 'logger_agent'", message)
        self.assertIn("'duration_s': 1.5", message)
        self.assertIn("'outputs_type': 'list'", message)
        self.assertIn("a" * 100, message)
        self.assertNotIn("a" * 101, message)
        self.assertNotIn("tokens", message)

    def test_log_call_with_tokens(self):
        self.agent.log_call({}, None, 0.1, tokens={"prompt": 3, "completion": 4})

        message = self.agent.logger.info.call_args.args[0]
        self.assertIn("'tokens': {'prompt': 3, 'completion': 4}", message)
        self.assertIn("'outputs_type': 'NoneType'", message)


if __name__ == "__main__":
    unittest.main()
