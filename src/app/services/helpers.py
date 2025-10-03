import json
import re
from functools import cache
from typing import List

import numpy
from langdetect import detect_langs
from qdrant_client.http.models import models

from src.app.models.documents import Document, JourneySectionType
from src.app.services.exceptions import LanguageNotSupportedError
from src.app.utils.decorators import log_time_and_error_sync
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


@log_time_and_error_sync
@cache
def convert_embedding_bytes(
    embeddings_byte: bytes, dtype=numpy.float32
) -> numpy.ndarray:
    """
    Converts a byte representation of embeddings to a numpy ndarray.
    Args:
        dtype: The desired data type of the output array. Default is numpy.float32. Only numpy types.
        embeddings_byte: The byte representation of the embeddings.
    Returns: A numpy ndarray of the embeddings.
    """
    if not isinstance(embeddings_byte, bytes):
        raise ValueError(
            f"Embedding must be of type bytes, received type: {type(embeddings_byte).__name__}"
        )
    return numpy.frombuffer(embeddings_byte, dtype=dtype)


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
        if primary_language not in ["en", "fr"]:
            raise LanguageNotSupportedError(
                "Language not supported", "LANG_NOT_SUPPORTED"
            )

        logger.info("Detected language: %s", primary_language)
        return primary_language

    except Exception:
        logger.error("api_error=LANG_NOT_DETECTED entry=%s", entry)
        raise LanguageNotSupportedError(
            "Error detecting language", "LANG_DETECTION_ERROR"
        )


def stringify_docs_content(docs: List[Document]) -> str:
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
        documents = "\n\n".join(
            base_article.format(
                number=i + 1,
                title=(doc.payload.document_title or "").strip(),
                content=(doc.payload.slice_content or "").strip(),
                url=(doc.payload.document_url or "").strip(),
            )
            for i, doc in enumerate(docs)
        )
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


def choose_readability_according_journey_section_type(
    sdg_doc_type: JourneySectionType,
) -> models.Range:
    """
    Choose the readability range according to the journey section type.
    1. Introduction: 60-100 (easier)
    2. Target: 0-60 (harder)
    Args:
        sdg_doc_type: The journey section type.

    Returns: The readability range for Qdrant search.

    """
    if sdg_doc_type == JourneySectionType.INTRODUCTION:
        readability_range = models.Range(
            gte=60,
            lte=100,
        )
    elif sdg_doc_type == JourneySectionType.TARGET:
        readability_range = models.Range(
            gte=0,
            lte=60,
        )
    else:
        raise NotImplementedError(
            f"Journey section type '{sdg_doc_type}' is not implemented."
        )
    return readability_range
