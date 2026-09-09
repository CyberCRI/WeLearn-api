import re
from functools import cache
from typing import Any, List, Optional, cast

import json_repair
import numpy
from fastapi import HTTPException
from json_repair import JSONReturnType
from langchain_core.messages import BaseMessage, ToolMessage
from langdetect import detect_langs
from qdrant_client.http.models import models
from welearn_database.data.models import EmbeddingModel

from src.app.models.collections import Collection
from src.app.models.documents import JourneySectionType
from src.app.services.sql_db.queries import get_embeddings_model_id_according_name
from src.app.shared.domain.exceptions import LanguageNotSupportedError
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
        return {}

    if isinstance(payload, dict):
        return payload

    for method_name in ("model_dump", "dict"):
        method = getattr(payload, method_name, None)
        if not callable(method):
            continue
        try:
            data = method()
            if isinstance(data, dict):
                return data
            return dict(cast(Any, data))
        except Exception:
            continue

    try:
        return dict(payload)
    except Exception:
        return {}


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

            # Only add article if at least one field is non-empty
            if title or content or url:
                articles.append(
                    base_article.format(
                        number=i + 1, title=title, content=content, url=url
                    )
                )

        documents = "\n\n".join(articles)
    except Exception as e:
        logger.exception("Error in stringify_docs_content: %s", e)
        return ""

    return documents.strip()


_CITATION_RE = re.compile(
    r'(?<!\[)(?<!target="_blank">)\[Doc\s*(\d+)\](?!\()', re.IGNORECASE
)


def linkify_missing_citations(text: str, docs: List[Any]) -> str:
    """
    Wraps any bare `[Doc N]` marker in `text` into a double-bracket Markdown link
    `[[Doc N]](URL)` for document N, using the URL from `docs[N-1]`. The double
    bracket is intentional: a plain Markdown link `[Doc N](URL)` renders with its
    brackets stripped (shows just "Doc N"), so the visible label must itself
    contain a literal "[Doc N]" for the brackets to survive rendering.

    Markers already wrapped this way, already a single-bracket Markdown link, or
    already wrapped in an HTML `<a>` tag, are left untouched. Safety net for when
    the LLM forgets to format a citation as a link itself.

    Only matches a single document number per marker — a combined citation like
    "[Docs 3 et 5]" is intentionally left as-is, since there's no single URL a
    two-document marker could safely resolve to.

    Args:
        text: The assembled answer text.
        docs: The full list of retrieved documents, 1-indexed by citation number.

    Returns:
        str: `text` with any missed citations auto-linked. Unresolvable markers
        (out-of-range N, or no docs) are left as-is.
    """
    if not text or not docs:
        return text

    def _url_for(n: int) -> Optional[str]:
        if not (1 <= n <= len(docs)):
            return None
        payload = normalize_payload(getattr(docs[n - 1], "payload", docs[n - 1]))
        return str(payload.get("document_url", "")).strip() or None

    def _replace(match: "re.Match[str]") -> str:
        n = int(match.group(1))
        url = _url_for(n)
        if not url:
            return match.group(0)
        return f"[[Doc {n}]]({url})"

    return _CITATION_RE.sub(_replace, text)


def latest_tool_docs(messages: List[BaseMessage]) -> Optional[List[Any]]:
    """
    Finds the most recent tool call's retrieved documents in a message list.

    Walks `messages` from the end, so a turn that made no new tool call still
    resolves to the last tool call's results instead of nothing, and a turn
    that did call the tool never picks up a stale, older call's results.

    Args:
        messages: The full conversation message list (may span many turns).

    Returns:
        The `artifact` of the most recent `ToolMessage` that has one, or
        None if no tool call with results is found.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "artifact", None):
            return msg.artifact
    return None


def extract_json_from_response(
    response: str,
) -> JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]] | str:
    """
    Extracts JSON object from a string response.
    Args:
        response (str): The string response containing JSON.
    Returns:
        dict: The extracted JSON object.
    """
    try:
        json_data = json_repair.loads(response)
        if json_data:
            return json_data
        else:
            raise ValueError("No JSON object found in the response")
    except Exception as e:
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


@log_time_and_error_sync
async def collection_and_model_id_according_lang(
    lang: str | None, sp
) -> tuple[Collection, list[EmbeddingModel]]:
    """
    Get the collection info and model id according to the language.
    Args:
        sp: The search service.
        lang: The language to get the collection info and model id for. If None, the default language is used wich is multilingual.

    Returns:
        A tuple of the collection info and model id.
    """
    if lang:
        collection_info = await sp.get_collection_by_language(lang)
    else:
        collection_info = await sp.get_collection_by_language()

    if not collection_info:
        raise HTTPException(
            status_code=404, detail=f"No collection found for language '{lang}'."
        )
    emb_models = get_embeddings_model_id_according_name(collection_info.model)
    if not emb_models:
        raise ValueError(
            f"Embedding model '{collection_info.model}' not found in the database."
        )
    return collection_info, emb_models
