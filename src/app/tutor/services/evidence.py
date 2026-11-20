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
from src.app.tutor.services.course_prompt import (
    connection_text,
    course_block,
    nth,
    resource_block,
)
from src.app.tutor.services.llm import LLM_FAILURES, DataBlock, structured_call
from src.app.tutor.services.resource_checks import (
    clean_quote,
    has_unsupported_figure,
    resource_text,
    verified_quote,
)
from src.app.tutor.services.text_match import normalise

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
    return connection_text(idea.title, idea.anchor, idea.why, idea.use)


def resource_blocks(docs: list[EnhancedDocumentSlice]) -> list[DataBlock]:
    return [resource_block(doc) for doc in docs]


# ---- checks ----


def has_link(doc: EnhancedDocumentSlice) -> bool:
    return bool(re.match(r"^https?://", doc.payload.document_url or ""))


def names_concept(sentence: str, terms: list[str]) -> bool:
    """The sentence contains one of the words given for the course's concept."""
    text = normalise(sentence)
    words = (normalise(term) for term in terms)
    return any(len(word) >= 3 and word in text for word in words)


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
        doc = nth(candidates, item.document)
        if doc is None or item.document in seen:
            continue
        seen.add(item.document)  # type: ignore[arg-type]
        # the key word used in another sense than the course's: not a resource for it
        if item.relevance is None or item.same_sense is False:
            continue
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
    except LLM_FAILURES:
        logger.warning("tutor_connection_evidence_unchecked idea=%s", idea.id)
        return unjudged.model_copy(update={"status": "unchecked"})
    return to_evidence(answer, idea, candidates)
