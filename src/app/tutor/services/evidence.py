"""
Syllabus v2: the WeLearn resources behind one connection
(`POST /tutor/course/connection_evidence`).

The idea's two searches are run on WeLearn, in every language, and their results merged. The
`selection` prompt then judges the connection and keeps a few resources. The checks below
keep only what the text proves: a quoted sentence must be in the resource, a resource is
direct support only when that sentence names the course's concept and the resource speaks
of sustainability, and no figure may be stated that the resources do not contain.
"""

import asyncio
import re

from fastapi import BackgroundTasks

from src.app.search.domain.models import EnhancedSearchQuery, SearchMethods
from src.app.search.services.search import SearchService, resolve_corpora
from src.app.shared.domain.documents import EnhancedDocumentSlice
from src.app.shared.domain.exceptions import UnusableLLMOutputError
from src.app.shared.utils.logger import logger as utils_logger
from src.app.tutor.domain.connections import (
    MAX_RESOURCES,
    ConfirmedCourse,
    ConnectionEvidenceResponse,
    ConnectionIdea,
    EvidencedResource,
    SelectionAnswer,
)
from src.app.tutor.domain.models import TutorLang
from src.app.tutor.services.connections import course_block
from src.app.tutor.services.course_reading import normalise, unify_characters
from src.app.tutor.services.llm import DataBlock, structured_call

logger = utils_logger(__name__)

# a few resources the teacher can weigh, from a few search results
MAX_CANDIDATES = 6
# as the client's searches: mostly relevance, a little diversity
RELEVANCE_FACTOR = 0.75


# ---- searching ----


async def search_candidates(
    idea: ConnectionIdea, search: SearchService, background_tasks: BackgroundTasks
) -> list[EnhancedDocumentSlice]:
    """
    The idea's searches on every visible corpus, in every language, merged by document
    (its best score kept), best first.
    """
    corpora = await resolve_corpora(None)
    results = await asyncio.gather(
        *(
            search.search_handler(
                qp=EnhancedSearchQuery(
                    query=query,
                    nb_results=MAX_CANDIDATES,
                    relevance_factor=RELEVANCE_FACTOR,
                    corpora=corpora,
                ),
                method=SearchMethods.BY_DOCUMENT,
                background_tasks=background_tasks,
                without_vectors=True,
            )
            for query in idea.queries
        )
    )
    found: dict[object, EnhancedDocumentSlice] = {}
    for docs in results:
        for doc in docs:
            known = found.get(doc.payload.document_id)
            if known is None or known.score < doc.score:
                found[doc.payload.document_id] = doc
    best = sorted(found.values(), key=lambda doc: doc.score, reverse=True)
    return best[:MAX_CANDIDATES]


# ---- what the model sees ----


def idea_block(idea: ConnectionIdea) -> str:
    lines = [
        f"Title: {idea.title}",
        f"Part of the course concerned: {idea.anchor or 'the whole course'}",
        f"Why: {idea.why}",
    ]
    if idea.use:
        lines.append(f"Use of the course's concept: {idea.use}")
    return "\n".join(lines)


def source_of(doc: EnhancedDocumentSlice) -> str:
    """Where a resource comes from, so its credibility can be judged."""
    details = doc.payload.document_details or {}
    parts = [
        doc.payload.document_corpus,
        details.get("publisher"),
        details.get("journal"),
        details.get("source"),
        doc.payload.document_lang,
    ]
    return " · ".join(str(part) for part in parts if part)


def resource_blocks(docs: list[EnhancedDocumentSlice]) -> list[DataBlock]:
    return [
        DataBlock(
            text=doc.payload.slice_content or "",
            attrs={"title": doc.payload.document_title, "source": source_of(doc)},
        )
        for doc in docs
    ]


# ---- checks ----

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


def has_link(doc: EnhancedDocumentSlice) -> bool:
    return bool(re.match(r"^https?://", doc.payload.document_url or ""))


def names_concept(sentence: str, terms: list[str]) -> bool:
    """The sentence contains one of the words given for the course's concept."""
    text = normalise(sentence)
    return any(len(normalise(t)) >= 3 and normalise(t) in text for t in terms)


# the model sometimes gives the sustainability words translated: common ones count too
SUSTAINABILITY_WORDS = [
    "sustainab",
    "durabl",
    "environment",
    "environnement",
    "climat",
    "ecolog",
    "écolog",
    "ecosystem",
    "écosystème",
    "biodivers",
    "pollut",
    "emission",
    "émission",
    "carbon",
    "greenhouse",
    "effetdeserre",
    "renewable",
    "renouvelable",
    "inequalit",
    "inégalit",
    "poverty",
    "pauvreté",
    "waste",
    "déchet",
    "circulareconomy",
    "économiecirculaire",
]


def speaks_of_sustainability(doc: EnhancedDocumentSlice, terms: list[str]) -> bool:
    text = resource_text([doc])
    return names_concept(text, terms) or any(w in text for w in SUSTAINABILITY_WORDS)


# a reason about the process ("…the connection chosen by the teacher, using the documents")
_PROCESS_WORDS = re.compile(
    r"\b(?:chosen by the teacher|the teacher chose|the (?:provided |cited )?documents?"
    r"|this connection|choisie? par l['’]enseignant|les documents|ce lien)\b",
    re.IGNORECASE,
)


def is_about_process(text: str) -> bool:
    return bool(_PROCESS_WORDS.search(text))


# ---- the selection, checked ----


def to_evidence(
    answer: SelectionAnswer,
    idea: ConnectionIdea,
    candidates: list[EnhancedDocumentSlice],
) -> ConnectionEvidenceResponse:
    """
    The judged resources that pass the checks. Direct support only when the model judges
    it so, its quoted sentence (verified) names the course's concept and the resource
    speaks of sustainability; otherwise background at most.
    """

    def response(status, resources=(), background=(), why=idea.why):
        return ConnectionEvidenceResponse(
            status=status,
            concept=answer.concept,
            why=why,
            resources=list(resources),
            background=list(background),
            candidate_count=len(candidates),
        )

    if answer.fit in ("analogy", "off_discipline"):
        return response("forced")
    if not answer.keep:
        return response("unsupported")

    seen: set[int] = set()
    direct: list[EvidencedResource] = []
    background: list[EvidencedResource] = []
    for item in answer.resources:
        n = item.document
        if n is None or not 1 <= n <= len(candidates) or n in seen:
            continue
        seen.add(n)
        # the key word used in another sense than the course's: not a resource for it
        if item.relevance is None or item.same_sense is False:
            continue
        doc = candidates[n - 1]
        evidence = verified_quote(clean_quote(item.evidence), doc)
        if not item.usefulness or not evidence or not has_link(doc):
            continue
        if has_unsupported_figure(item.usefulness, [doc]):
            continue
        is_direct = (
            item.relevance == "direct"
            and item.same_sense is True
            and item.establishes is True
            and names_concept(evidence, item.concept_terms)
            and speaks_of_sustainability(doc, item.sustainability_terms)
        )
        resource = EvidencedResource(
            document=doc, usefulness=item.usefulness, evidence=evidence
        )
        if len(direct) + len(background) < MAX_RESOURCES:
            (direct if is_direct else background).append(resource)

    if not direct:
        return response("unsupported", background=background)
    kept = [resource.document for resource in direct]
    why = answer.why
    is_usable_why = (
        why and not has_unsupported_figure(why, kept) and not is_about_process(why)
    )
    return response("supported", direct, background, why if is_usable_why else idea.why)


async def find_connection_evidence(
    course: ConfirmedCourse,
    idea: ConnectionIdea,
    lang: TutorLang,
    search: SearchService,
    background_tasks: BackgroundTasks,
) -> ConnectionEvidenceResponse:
    """
    The resources for one idea. Without a search result there is nothing to judge
    (unsupported); an answer still unusable after a retry leaves the idea unchecked.
    """
    candidates = await search_candidates(idea, search, background_tasks)
    unjudged = ConnectionEvidenceResponse(
        status="unsupported",
        concept=None,
        why=idea.why,
        resources=[],
        background=[],
        candidate_count=len(candidates),
    )
    if not candidates:
        return unjudged
    try:
        answer = await structured_call(
            "selection",
            lang,
            SelectionAnswer,
            {
                "course": course_block(course),
                "idea": idea_block(idea),
                "resource": resource_blocks(candidates),
            },
            {"count": len(candidates)},
        )
    except UnusableLLMOutputError:
        logger.warning("tutor_connection_evidence_unchecked idea=%s", idea.id)
        return unjudged.model_copy(update={"status": "unchecked"})
    return to_evidence(answer, idea, candidates)
