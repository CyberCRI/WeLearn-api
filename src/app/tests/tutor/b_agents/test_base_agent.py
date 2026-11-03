import unittest
from unittest.mock import MagicMock, patch

from src.app.tutor.services.b_agents.base import AgentError, BaseAgent

SLEEP_PATH = "src.app.tutor.services.b_agents.base.time.sleep"


class Payload:
    def __init__(self, text: str):
        self.text = text


class TestBaseAgentRetry(unittest.TestCase):
    def setUp(self):
        self.agent = BaseAgent(name="test_agent", max_retries=3)

    def test_success_first_try_tracks_usage(self):
        func = MagicMock(return_value="x" * 40)

        with patch(SLEEP_PATH) as mock_sleep:
            result = self.agent.with_retry(func, "y" * 40, key="z" * 40)

        self.assertEqual(result, "x" * 40)
        func.assert_called_once_with("y" * 40, key="z" * 40)
        mock_sleep.assert_not_called()
        self.assertEqual(self.agent.call_count, 1)
        # (40 + 40 + 40) / 4 * 1.5
        self.assertEqual(self.agent.total_tokens, 45)

    def test_retries_then_succeeds_with_exponential_backoff(self):
        func = MagicMock(side_effect=[ValueError("e1"), ValueError("e2"), "ok"])

        with patch(SLEEP_PATH) as mock_sleep:
            result = self.agent.with_retry(func)

        self.assertEqual(result, "ok")
        self.assertEqual(func.call_count, 3)
        self.assertEqual([c.args[0] for c in mock_sleep.call_args_list], [1, 2])
        self.assertEqual(self.agent.call_count, 1)

    def test_all_retries_fail_raises_agent_error(self):
        err = RuntimeError("down")
        func = MagicMock(side_effect=err)

        with patch(SLEEP_PATH) as mock_sleep:
            with self.assertRaises(AgentError) as ctx:
                self.agent.with_retry(func)

        self.assertEqual(func.call_count, 3)
        # no sleep after the last attempt
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertIs(ctx.exception.__cause__, err)
        self.assertIn("test_agent failed after 3 attempts: down", str(ctx.exception))
        self.assertEqual(self.agent.call_count, 0)
        self.assertEqual(self.agent.total_tokens, 0)

    def test_single_retry_never_sleeps(self):
        agent = BaseAgent(name="once", max_retries=1)

        with patch(SLEEP_PATH) as mock_sleep:
            with self.assertRaises(AgentError):
                agent.with_retry(MagicMock(side_effect=ValueError("x")))

        mock_sleep.assert_not_called()


class TestEstimateTokens(unittest.TestCase):
    def setUp(self):
        self.agent = BaseAgent(name="tok")

    def test_strings(self):
        self.assertEqual(self.agent._estimate_tokens(("abcd",), {}, "abcd"), 3)

    def test_nested_containers_and_objects(self):
        args = (["ab", ("cd", "ef")], {"k": "gh"})
        kwargs = {"obj": Payload("ijkl")}
        # 2 + 2 + 2 + 2 + 4 = 12 input chars; result 42 -> len("42") == 2
        self.assertEqual(self.agent._estimate_tokens(args, kwargs, 42), 5)

    def test_empty(self):
        self.assertEqual(self.agent._estimate_tokens((), {}, ""), 0)


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
