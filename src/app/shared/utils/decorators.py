import functools
import inspect
import time

from src.app.shared.utils.db_errors import describe_error
from src.app.shared.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


def _log_args(kwargs: dict) -> str:
    """Render the kwargs worth logging (collection and model names)."""
    log_args = ""
    for key, value in kwargs.items():
        if "collection_" in key or key == "model":
            log_args += f"{key}={value}, "
    return log_args


def _log_success(func, kwargs: dict, start_time: float) -> None:
    logger.info(
        _log_args(kwargs) + "method=%s latency=%s",
        func.__name__,
        round(time.perf_counter() - start_time, 2),
    )
    logger.debug("finishing method=%s", func.__name__)


def _log_error(func, error: Exception) -> None:
    logger.error("method=%s api_error=%s", func.__name__, describe_error(error))


def log_time_and_error(func):
    """Log the latency of ``func`` and any exception it raises, then re-raise.

    Works on both regular and ``async def`` functions: coroutine functions get
    an async wrapper that awaits them, so the latency covers the real work and
    exceptions raised while awaiting are logged.

    Generator and async generator functions are rejected: their body only runs
    while being iterated, after the wrapper has returned, so the latency and
    error logging would be meaningless.
    """
    if inspect.isasyncgenfunction(func) or inspect.isgeneratorfunction(func):
        raise TypeError(
            f"log_time_and_error does not support generator functions "
            f"({func.__qualname__})"
        )

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug("starting method=%s", func.__name__)
            try:
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                _log_success(func, kwargs, start_time)
                return result
            except Exception as e:
                _log_error(func, e)
                raise

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("starting method=%s", func.__name__)
        try:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            _log_success(func, kwargs, start_time)
            return result
        except Exception as e:
            _log_error(func, e)
            raise

    return wrapper


def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instance
