import re
from typing import Any, List, Optional, cast

from langchain_core.messages import BaseMessage, ToolMessage

from src.app.shared.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


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
