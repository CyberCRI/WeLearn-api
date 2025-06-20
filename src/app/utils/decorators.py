import functools
import time

from src.app.utils.logger import logger

logger = logger(__name__)


def log_time_and_error(func):
    functools.wraps(func)

    async def wrapper(*args, **kwargs):
        logger.debug("starting method=%s", func.__name__)
        try:
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()

            log_args = ""
            for key, value in kwargs.items():
                if "collection_" in key or key == "model":
                    log_args += f"{key}={value}, "

            logger.info(
                log_args + "method=%s latency=%s",
                func.__name__,
                round(end_time - start_time, 2),
            )

            logger.debug("finishing method=%s", func.__name__)
            return result
        except Exception as e:
            logger.error("method=%s api_error=%s", func.__name__, e)
            raise e

    return wrapper


def log_time_and_error_sync(func):
    functools.wraps(func)

    def wrapper(*args, **kwargs):
        logger.debug("starting method=%s", func.__name__)
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            log_args = ""
            for key, value in kwargs.items():
                if "collection_" in key or key == "model":
                    log_args += f"{key}={value}, "

            logger.info(
                log_args + "method=%s latency=%s",
                func.__name__,
                round(end_time - start_time, 2),
            )
            logger.debug("finishing method=%s", func.__name__)
            return result
        except Exception as e:
            logger.error("method=%s api_error=%s", func.__name__, e)
            raise e

    return wrapper


def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instance
