"""
Client of the SDG classifier (Aurora, VU Amsterdam). The API calls it rather than the
browser, so the classifier gets the text but not the user's IP address or browser.
"""

import asyncio

import httpx
from pydantic import BaseModel, Field, ValidationError

from src.app.shared.domain.exceptions import SDGClassifierUnavailableError
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils

logger = logger_utils(__name__)
settings = get_settings()


class ClassifierSDG(BaseModel):
    # the classifier sends the goal number as a string ("13")
    code: int = Field(ge=1, le=17)


class SDGPrediction(BaseModel):
    prediction: float
    sdg: ClassifierSDG


class ClassifierAnswer(BaseModel):
    predictions: list[SDGPrediction]


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=settings.SDG_CLASSIFIER_TIMEOUT_SECONDS)


async def get_sdg_predictions(text: str) -> list[SDGPrediction]:
    """The classifier's confidence for each SDG, in its order."""
    try:
        # httpx's timeout bounds each read, this one the whole call
        async with asyncio.timeout(settings.SDG_CLASSIFIER_TIMEOUT_SECONDS):
            async with _http_client() as client:
                response = await client.post(
                    settings.SDG_CLASSIFIER_URL, json={"text": text}
                )
        response.raise_for_status()
        return ClassifierAnswer.model_validate_json(response.content).predictions
    except (TimeoutError, httpx.HTTPError, ValidationError) as e:
        # never log the text: it is the user's query
        logger.warning(
            "sdg_classifier_unavailable error=%s status=%s",
            type(e).__name__,
            getattr(getattr(e, "response", None), "status_code", None),
        )
        raise SDGClassifierUnavailableError() from e
