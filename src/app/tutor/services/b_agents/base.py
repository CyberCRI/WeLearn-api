"""
Base agent class with retry logic and error handling
"""

import logging
import time
from typing import Any, Callable, Dict, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AgentError(Exception):
    """Custom exception for agent errors"""

    pass


class BaseAgent:
    """Base class for all agents with common functionality"""

    def __init__(self, name: str, max_retries: int = 3):
        """
        Initialize agent

        Args:
            name: Agent name for logging
            max_retries: Maximum number of retry attempts
        """
        self.name = name
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"agent.{name}")
        self.total_tokens = 0  # Track total tokens used by this agent
        self.call_count = 0  # Track number of calls

    def with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic and token tracking

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            AgentError: If all retries fail
        """
        attempt = 1
        last_error = None

        while attempt <= self.max_retries:
            try:
                start_time = time.time()
                self.logger.info(f"Attempt {attempt}/{self.max_retries}")

                result = func(*args, **kwargs)

                duration = time.time() - start_time
                self.logger.info(f"Success in {duration:.2f}s")

                # Estimate token usage (rough approximation)
                # BAML doesn't expose token counts directly, so we estimatebased on input/output sizes
                # 1 token ≈ 4 characters for English text
                estimated_tokens = self._estimate_tokens(args, kwargs, result)
                self.total_tokens += estimated_tokens
                self.call_count += 1

                self.logger.info(f"Estimated tokens for this call: {estimated_tokens}")
                self.logger.info(
                    f"Total tokens used by {self.name}: {self.total_tokens}"
                )

                return result

            except Exception as e:
                last_error = e
                duration = time.time() - start_time

                self.logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed after {duration:.2f}s: {str(e)}"
                )

                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** (attempt - 1)
                    self.logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

                attempt += 1

        # All retries failed
        error_msg = (
            f"{self.name} failed after {self.max_retries} attempts: {str(last_error)}"
        )
        self.logger.error(error_msg)
        raise AgentError(error_msg) from last_error

    def _estimate_tokens(self, args, kwargs, result) -> int:
        """
        Estimate token usage based on input/output sizes

        Rough approximation: 1 token ≈ 4 characters

        Args:
            args: Function args
            kwargs: Function kwargs
            result: Function result

        Returns:
            Estimated token count
        """

        def count_chars(obj):
            """Recursively count characters in an object"""
            if isinstance(obj, str):
                return len(obj)
            elif isinstance(obj, (list, tuple)):
                return sum(count_chars(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(count_chars(v) for v in obj.values())
            elif hasattr(obj, "__dict__"):
                return count_chars(obj.__dict__)
            else:
                return len(str(obj))

        input_chars = sum(count_chars(arg) for arg in args)
        input_chars += sum(count_chars(v) for v in kwargs.values())
        output_chars = count_chars(result)

        # Total chars / 4 ≈ tokens
        # Multiply by 1.5 to account for prompt structure, formatting, etc. --> TODO:NOT SURE ABOUT THIS
        estimated = int((input_chars + output_chars) / 4 * 1.5)

        return estimated

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
