import os

from src.app.core.config import Settings
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


def configure_langsmith_tracing(settings: Settings) -> None:
    """Configure LangSmith tracing for LangChain-based model calls."""

    is_enabled = settings.LANGSMITH_TRACING_ENABLED

    os.environ["LANGSMITH_TRACING"] = "true" if is_enabled else "false"

    if not is_enabled:
        logger.info("langsmith_tracing_enabled=false")
        return

    if settings.LANGSMITH_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    else:
        logger.warning(
            "langsmith_tracing_enabled=true but LANGSMITH_PROJECT is missing"
        )

    if settings.LANGSMITH_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    else:
        logger.warning(
            "langsmith_tracing_enabled=true but LANGSMITH_ENDPOINT is missing"
        )

    if settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    else:
        logger.warning(
            "langsmith_tracing_enabled=true but LANGSMITH_API_KEY is missing"
        )

    logger.info(
        "langsmith_tracing_enabled=true project=%s endpoint=%s",
        settings.LANGSMITH_PROJECT,
        settings.LANGSMITH_ENDPOINT,
    )
