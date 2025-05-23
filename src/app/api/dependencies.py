import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.app.core import config
from src.app.utils.logger import logger as logger_utils

USE_CACHED_SETTINGS = os.getenv("USE_CACHED_SETTINGS", "True") == "True"
logger = logger_utils(__name__)


@lru_cache()
def get_cached_settings():
    """_summary_

    Returns:
        _type_: _description_
    """

    return config.Settings()


def get_settings():
    """_summary_

    Returns:
        _type_: _description_
    """

    if USE_CACHED_SETTINGS:
        logger.debug("cached_setting=true")
        return get_cached_settings()
    else:
        return config.Settings()


ConfigDepend = Annotated[config.Settings, Depends(get_settings)]
