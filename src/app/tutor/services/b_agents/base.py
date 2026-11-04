"""
Base agent class with retry logic and error handling
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, TypeVar

from baml_py.errors import BamlClientFinishReasonError, BamlValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Errors worth re-running the whole BAML function for: the request succeeded but the
# LLM output could not be parsed into the expected type, or was cut short. A new
# sampling usually fixes it.
#
# Transport errors (connection errors, timeouts, HTTP 429/5xx) are deliberately NOT
# listed: the `MistralSmall` client in `src/app/baml_src/clients.baml` already retries
# them with its `retry_policy`. Retrying them here too would multiply the attempts
# (3 HTTP calls x max_retries), each one bounded by a long request timeout. They reach
# the caller unchanged once BAML gives up, like any other non-transient error.
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    BamlValidationError,
    BamlClientFinishReasonError,
)


class AgentError(Exception):
    """Raised when an agent still fails after all its retry attempts"""

    pass


class BaseAgent:
    """Base class for all agents with common functionality"""

    def __init__(self, name: str, max_retries: int = 2):
        """
        Initialize agent

        Args:
            name: Agent name for logging
            max_retries: Maximum number of attempts for output (parse) failures
        """
        self.name = name
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"agent.{name}")

    async def with_retry(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Await ``func(*args, **kwargs)``, retrying on LLM output failures.

        Two retry layers exist, with distinct jobs:

        - the BAML client (``retry_policy`` in ``clients.baml``) retries the HTTP
          request itself: connection errors, timeouts, 429 and 5xx (BAML does not
          filter by status, so other 4xx are retried there too);
        - this method re-runs the whole call only when the LLM answered but its
          output was unusable (``RETRYABLE_ERRORS``), up to ``max_retries`` attempts
          with exponential backoff (1s, 2s, ...) that does not block the event loop.

        Any other exception (transport errors already retried by BAML, HTTP 4xx,
        programming errors) propagates immediately and unchanged.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            AgentError: If every attempt failed with a retryable error
        """
        last_error: BaseException | None = None

        for attempt in range(1, self.max_retries + 1):
            start_time = time.monotonic()
            try:
                result = await func(*args, **kwargs)
            except RETRYABLE_ERRORS as e:
                last_error = e
                self.logger.warning(
                    "Attempt %s/%s failed after %.2fs: %s",
                    attempt,
                    self.max_retries,
                    time.monotonic() - start_time,
                    type(e).__name__,
                )
                if attempt < self.max_retries:
                    wait_time = 2 ** (attempt - 1)
                    self.logger.info("Retrying in %ss...", wait_time)
                    await asyncio.sleep(wait_time)
                continue

            self.logger.info(
                "Success on attempt %s/%s in %.2fs",
                attempt,
                self.max_retries,
                time.monotonic() - start_time,
            )
            return result

        error_msg = (
            f"{self.name} failed after {self.max_retries} attempts: "
            f"{type(last_error).__name__}"
        )
        self.logger.error(error_msg)
        raise AgentError(error_msg) from last_error

    def log_call(
        self,
        inputs: Dict[str, Any],
        outputs: Any,
        duration: float,
        tokens: Dict[str, int] = None,
    ):
        """
        Log agent call details

        Args:
            inputs: Input parameters
            outputs: Output result
            duration: Execution duration in seconds
            tokens: Token usage (prompt, completion)
        """
        log_data = {
            "agent": self.name,
            "duration_s": duration,
            "inputs_preview": {k: str(v)[:100] for k, v in inputs.items()},
            "outputs_type": type(outputs).__name__,
        }

        if tokens:
            log_data["tokens"] = tokens

        self.logger.info(f"Call completed: {log_data}")
