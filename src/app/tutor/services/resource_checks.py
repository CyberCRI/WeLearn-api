"""
Syllabus v2: what the model says, checked against the WeLearn resources it was given:
a quote must be in its resource, a figure in one of them.
"""

import re

from src.app.shared.domain.documents import EnhancedDocumentSlice
from src.app.tutor.services.text_match import normalise, unify_characters

# figures stated (percentages, decimal numbers, amounts in millions or billions, currencies)
_FIGURES = re.compile(
    r"\d+(?:[.,]\d+)?\s?(?:%|milliards?|millions?|billions?|bn|€|\$|£)"
    r"|[€$£]\s?\d+(?:[.,]\d+)?|\d+[.,]\d+",
    re.IGNORECASE,
)


def _text_of(doc: EnhancedDocumentSlice) -> str:
    return f"{doc.payload.document_title} {doc.payload.slice_content or ''}"


def resource_text(docs: list[EnhancedDocumentSlice]) -> str:
    return normalise(" ".join(_text_of(doc) for doc in docs))


def has_unsupported_figure(text: str, docs: list[EnhancedDocumentSlice]) -> bool:
    """A figure in `text` that the resources do not contain."""
    source = resource_text(docs)
    return any(normalise(figure) not in source for figure in _FIGURES.findall(text))


# text extraction may put a page header or number inside a sentence of a pdf. A quote of
# at least MIN_LOOSE_WORDS words is still found with up to MAX_INSERTED_WORDS such words
# inside it; a shorter quote, with so few words of its own, must be found as it is
MIN_LOOSE_WORDS = 8
MAX_INSERTED_WORDS = 8
# a sentence kept from a quote stitched from several
MIN_SENTENCE = 40

# a word cut by a hyphen at the end of a pdf line, the line break read as a space
_CUT_WORD = re.compile(r"([^\W\d_])-\s+([^\W\d_])")
_NOT_A_WORD = re.compile(r"[\W_]+")


def words_of(text: str) -> list[str]:
    """
    The words of a text, its characters unified as `normalise` does (so both checks of
    `is_quoted_in` read characters the same way); "diffi- cult" is one word.
    """
    text = _CUT_WORD.sub(r"\1\2", unify_characters(text))
    return [word for word in _NOT_A_WORD.split(text) if word]


def _in_order_with_few_insertions(
    doc_words: list[str], start: int, quote_words: list[str]
) -> bool:
    """
    The quote's words appear in the document from `start` on, in order, with at most
    MAX_INSERTED_WORDS other words between them in total.

    Each quote word is matched at its first appearance: that ends the match as early as
    possible, so it inserts the fewest words.
    """
    found = inserted = 0
    for word in doc_words[start:]:
        if word == quote_words[found]:
            found += 1
            if found == len(quote_words):
                return True
        else:
            inserted += 1
            if inserted > MAX_INSERTED_WORDS:
                return False
    return False


def is_quoted_in(quote: str, doc: EnhancedDocumentSlice) -> bool:
    """
    The quote is in the resource, with only what text extraction adds to a pdf:

    1. the quote is in the text once case, spaces, line breaks, quote and dash styles are
       set aside (`normalise`). Word boundaries are set aside too, a trade-off that is
       harmless for a sentence. Hyphens are tried without as well: a pdf cuts "difficult"
       into "diffi- cult", which reads "diffi-cult" without spaces;
    2. otherwise, for a quote of at least MIN_LOOSE_WORDS words: every word of the quote,
       in order, with at most MAX_INSERTED_WORDS other words between them in total (a page
       header inside the sentence).

    Every word of the quote must be in the resource; the reverse is not checked. So a
    quote that leaves out a few words of the resource's sentence is still found: that is
    how a page header is skipped, and the price of it. A word changed by the model is
    caught, unless the quote's own word comes a few words further on in the resource.
    """
    doc_compact = resource_text([doc])
    quote_compact = normalise(quote)
    if not quote_compact:
        return False
    if quote_compact in doc_compact:
        return True
    if quote_compact.replace("-", "") in doc_compact.replace("-", ""):
        return True

    quote_words = words_of(quote)
    if len(quote_words) < MIN_LOOSE_WORDS:
        return False
    doc_words = words_of(_text_of(doc))
    return any(
        _in_order_with_few_insertions(doc_words, start, quote_words)
        for start, word in enumerate(doc_words)
        if word == quote_words[0]
    )


_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
_ELLIPSIS = re.compile(r"^\[(\.\.\.|…)\]\s*")


def verified_quote(quote: str | None, doc: EnhancedDocumentSlice) -> str | None:
    """
    The part of a quote really in the resource: all of it, or else the first of its
    sentences that is (a quote stitched from sentences far apart keeps only a true one).
    None when none is.
    """
    if not quote:
        return None
    if is_quoted_in(quote, doc):
        return quote
    sentences = [
        _ELLIPSIS.sub("", part)
        for part in _SENTENCE_END.split(quote)
        if len(_ELLIPSIS.sub("", part)) >= MIN_SENTENCE
    ]
    if len(sentences) < 2:
        return None
    return next((part for part in sentences if is_quoted_in(part, doc)), None)


_QUOTE_MARKS = re.compile(r'^[«"“\s]+|[»"”\s]+[.;,]?\s*$')


def clean_quote(value: str | None) -> str | None:
    """The quote without the quotation marks around it."""
    return _QUOTE_MARKS.sub("", value or "") or None
