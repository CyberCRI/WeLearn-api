import json
import re
from functools import cache
from typing import Any, List

from langdetect import detect_langs  # type: ignore

from src.app.services.exceptions import LanguageNotSupportedError
from src.app.utils.decorators import log_time_and_error_sync
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


@log_time_and_error_sync
@cache
def detect_language_from_entry(entry: str) -> str:
    """
    Detects the language from the given text entry.

    Args:
        entry (str): The text entry to detect the language from.

    Returns:
        str: The detected language code.

    Raises:
        LanguageNotSupportedError: If no language is detected or the detected language is not supported.
    """
    try:
        detected_languages = detect_langs(entry)
        if not detected_languages:
            raise LanguageNotSupportedError("No language detected", "LANG_NOT_DETECTED")

        primary_language = detected_languages[0].lang

        logger.info("Detected language: %s", primary_language)
        return primary_language

    except Exception:
        logger.error("api_error=LANG_NOT_DETECTED entry=%s", entry)
        raise LanguageNotSupportedError(
            "Error detecting language", "LANG_DETECTION_ERROR"
        )


def normalize_payload(payload: Any) -> dict:
    # Normalize payload to a dict
    if payload is None:
        payload = {}
    elif hasattr(payload, "dict") and callable(getattr(payload, "dict")):
        try:
            payload = payload.dict()
        except Exception:
            payload = dict(payload)
    elif not isinstance(payload, dict):
        try:
            payload = dict(payload)
        except Exception:
            payload = {}
    return payload


def stringify_docs_content(docs: List[Any]) -> str:
    """
    Creates a string from a list of documents.
    If document_title and slice_content are present,
    it is used to create a prompt.

    Args:
        docs (List[Document]): List of documents from search.

    Returns:
        str: A formatted string containing document details.
    """

    base_article = (
        """<article>\nDoc {number}: {title}\n{content}\n\nurl:{url}</article>"""
    )
    try:
        articles: list[str] = []
        for i, doc in enumerate(docs):
            payload = getattr(doc, "payload", {})

            payload = normalize_payload(payload)

            title = str(payload.get("document_title", "")).strip()
            content = str(payload.get("slice_content", "")).strip()
            url = str(payload.get("document_url", "")).strip()

            articles.append(
                base_article.format(number=i + 1, title=title, content=content, url=url)
            )

        documents = "\n\n".join(articles)
    except Exception as e:
        logger.error("Error in stringify_docs_content: %s", e)
        return ""

    return documents.strip()


def extract_json_from_response(response: str) -> dict:
    """
    Extracts JSON object from a string response.
    Args:
        response (str): The string response containing JSON.
    Returns:
        dict: The extracted JSON object.
    """
    try:
        json_data = re.search(r"\{.*\}", response, re.DOTALL)
        if json_data:
            return json.loads(json_data.group())
        else:
            raise ValueError("No JSON object found in the response")
    except json.JSONDecodeError as e:
        logger.error("api_error=json_decode_error, response=%s", response)
        raise e
