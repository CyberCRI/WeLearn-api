"""
Syllabus v2: one concrete change to the course for a connection the teacher chose
(`POST /tutor/course/adaptation`).

1. `adaptation` proposes the change from the connection's resources;
2. `verification` reads its claims again, one by one, against those resources (asked apart,
   so the model reads the change as a text to check, not as its own work);
3. if claims were not established or wrong, the change is asked again once without them,
   and the second one replaces the first only when it could be checked too;
4. the support is capped by what the check found: never raised, only lowered.

At most 4 steps; each asks the model again once when its answer can't be used, so 8 LLM
calls at worst. The change's text is left as the model wrote it, for the teacher to judge;
what the code checks shows in `support` and `checks`.
"""

import re
from dataclasses import dataclass
from typing import Literal

from src.app.shared.domain.documents import EnhancedDocumentSlice
from src.app.shared.utils.logger import logger as utils_logger
from src.app.tutor.domain.adaptation import (
    AdaptationAnswer,
    AdaptationChecks,
    AdaptationGuidance,
    AdaptationProposal,
    AdaptationResource,
    AdaptationSupport,
    ChosenConnection,
    GreenCompLink,
    GuidanceAnswer,
    VerificationAnswer,
)
from src.app.tutor.domain.connections import ConfirmedCourse, EvidencedResource
from src.app.tutor.domain.greencomp import GREENCOMP_IDS, greencomp_for_prompt
from src.app.tutor.domain.models import TutorLang
from src.app.tutor.services.connections import course_parts, part_of_course
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
    words_of,
)
from src.app.tutor.services.text_match import names_each_other, normalise

logger = utils_logger(__name__)

# the teacher reads a few problems, not a list
MAX_PROBLEMS = 4


@dataclass(frozen=True)
class RoledResource:
    """A resource of the connection with its role: direct support, or context only."""

    resource: EvidencedResource
    role: Literal["direct", "context"]


def roled_resources(connection: ChosenConnection) -> list[RoledResource]:
    """Its direct resources first, then those that only give context."""
    return [RoledResource(r, "direct") for r in connection.resources] + [
        RoledResource(r, "context") for r in connection.background
    ]


# ---- what the model sees ----


def connection_block(connection: ChosenConnection) -> str:
    return connection_text(connection.title, connection.anchor, connection.why)


def resource_blocks(resources: list[RoledResource]) -> list[DataBlock]:
    return [resource_block(r.resource.document, r.role) for r in resources]


def problems_block(problems: list[str]) -> str:
    return "\n".join(f"- {problem}" for problem in problems)


# ---- the proposal, checked ----


def tidy_note(text: str) -> str:
    """A note as a sentence: capital first letter, one closing punctuation mark."""
    trimmed = re.sub(r"[\s:;,]+$", "", re.sub(r"\s+", " ", text).strip())
    if not trimmed:
        return trimmed
    capitalised = trimmed[0].upper() + trimmed[1:]
    return capitalised if re.search(r"[.!?…]$", capitalised) else f"{capitalised}."


def place_of_change(
    answer: AdaptationAnswer, course: ConfirmedCourse, anchor: str | None
) -> str | None:
    """
    The course part the change touches, as the teacher wrote it. An addition to a topic or
    a teaching adjustment stays with the part its connection is tied to: the model's part
    is kept only when it is that part or the topic that contains it, and the connection's
    part is used when the model names none. A revised objective or an assessment change
    never borrows the connection's part.
    """
    parts = course_parts(course)
    named = part_of_course(answer.original, parts)
    if answer.kind not in ("topic", "teaching"):
        return named
    linked = part_of_course(anchor, parts)
    if named is None or linked is None:
        return linked or named
    is_with_link = names_each_other(named, linked) or any(
        topic.title == named
        and any(names_each_other(sub, linked) for sub in topic.subtopics)
        for topic in course.topics
    )
    return named if is_with_link else linked


def _cited_resources(
    answer: AdaptationAnswer, resources: list[RoledResource]
) -> list[AdaptationResource]:
    """The resources the answer cites, each with its quote when found word for word in it."""
    quotes: dict[int, str] = {}
    for item in answer.evidence:
        given = nth(resources, item.document)
        if given is None or item.document in quotes:
            continue
        quote = verified_quote(clean_quote(item.quote), given.resource.document)
        if quote:
            quotes[item.document] = quote  # type: ignore[index]

    cited_numbers = [
        n
        for n in dict.fromkeys([*answer.documents, *quotes])
        if nth(resources, n) is not None
    ]
    return [
        AdaptationResource(
            **resources[n - 1].resource.model_dump(),
            role=resources[n - 1].role,
            quote=quotes.get(n),
        )
        for n in cited_numbers
    ]


def _capped_support(
    claimed: AdaptationSupport | None, cited: list[AdaptationResource]
) -> AdaptationSupport:
    """ "strong" needs a quote of a direct resource; a change citing nothing has none."""
    if not cited:
        return "none"
    is_directly_quoted = any(r.role == "direct" and r.quote for r in cited)
    if claimed == "strong" and not is_directly_quoted:
        return "partial"
    return claimed or "none"


def _checked_guidance(
    guidance: GuidanceAnswer, cited_docs: list[EnhancedDocumentSlice]
) -> AdaptationGuidance:
    """The optional notes, each dropped when it states a figure the resources lack."""

    def note(text: str | None) -> str | None:
        if not text or has_unsupported_figure(text, cited_docs):
            return None
        return tidy_note(text) or None

    competence = guidance.greencomp
    how = note(competence.how) if competence else None
    greencomp = (
        GreenCompLink(id=competence.id, how=how)  # type: ignore[arg-type]
        if competence and competence.id in GREENCOMP_IDS and how
        else None
    )
    return AdaptationGuidance(
        learning=note(guidance.learning),
        greencomp=greencomp,
        teaching=note(guidance.teaching),
        check=note(guidance.check),
    )


def to_proposal(
    answer: AdaptationAnswer,
    course: ConfirmedCourse,
    connection: ChosenConnection,
    resources: list[RoledResource],
) -> AdaptationProposal | None:
    """
    The proposal with what the code can check, or None when it can't be used (asked again):
    no change or reason, an unknown kind, a revised objective the course does not have, or
    a figure the cited resources do not contain.

    "strong" support needs a sentence of a direct resource found word for word in it; a
    change that cites no resource is never supported.
    """
    if not answer.kind or not answer.suggested or not answer.why:
        return None
    original = place_of_change(answer, course, connection.anchor)
    # a revised objective revises one the course has, never a new one
    if answer.kind == "objective" and original is None:
        return None

    cited = _cited_resources(answer, resources)
    cited_docs = [resource.document for resource in cited]
    if has_unsupported_figure(f"{answer.suggested} {answer.why}", cited_docs):
        return None

    return AdaptationProposal(
        kind=answer.kind,
        original=original,
        suggested=answer.suggested,
        why=answer.why,
        support=_capped_support(answer.support, cited),
        resources=cited,
        guidance=_checked_guidance(answer.guidance, cited_docs),
        checks=None,
    )


# ---- the check of its claims ----

# a claim's content words, to tell a claim of the change from one taken from the documents
MIN_CLAIM_WORD = 5


def restates_change(claim: str, change: str) -> bool:
    """At least half of the claim's content words are in the change's text."""
    text = " ".join(words_of(change))
    words = [word for word in words_of(claim) if len(word) >= MIN_CLAIM_WORD]
    return bool(words) and 2 * sum(word in text for word in words) >= len(words)


@dataclass(frozen=True)
class Verification:
    problems: list[str]
    has_incorrect: bool
    has_unestablished: bool
    is_forced: bool


def to_verification(
    answer: VerificationAnswer, resources: list[RoledResource], change: str
) -> Verification:
    """
    What the check found. Only the change's own claims count. A claim the model calls
    established counts only with a sentence of a direct resource found word for word in
    it; otherwise it is not established.
    """
    problems: list[str] = []
    has_incorrect = has_unestablished = False
    for item in answer.claims:
        if not item.claim or not restates_change(item.claim, change):
            continue
        resource = nth(resources, item.document)
        is_backed = (
            item.status == "established"
            and resource is not None
            and resource.role == "direct"
            and verified_quote(clean_quote(item.quote), resource.resource.document)
            is not None
        )
        if is_backed:
            continue
        if item.status == "incorrect":
            has_incorrect = True
        else:
            has_unestablished = True
        problems.append(tidy_note(item.problem or item.claim))
    is_forced = answer.fit == "forced"
    if is_forced and answer.fit_problem:
        problems.insert(0, tidy_note(answer.fit_problem))
    return Verification(
        problems=problems[:MAX_PROBLEMS],
        has_incorrect=has_incorrect,
        has_unestablished=has_unestablished,
        is_forced=is_forced,
    )


# ---- names the change states ----

# sustainability's own acronyms, the most common formulas, capitalised common nouns
COMMON_NAMES = {
    "sdg",
    "sdgs",
    "odd",
    "greencomp",
    "co2",
    "h2o",
    "o2",
    "ch4",
    "earth",
    "sun",
    "moon",
    "terre",
    "soleil",
    "lune",
}
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# a sentence may end inside quotation marks ("…?' Use")
_SENTENCE = re.compile(r"(?<=[.!?:;])[\"'’”»)]*\s+")
_WORD = re.compile(r"[^\W_][\w'’-]*")
# a name in another language than the resource ("Australie", "Australia") is found by its
# stem: all but its last three letters, five at least
MIN_STEM = 5


def plain_digits(text: str) -> str:
    """ "CO₂" is "CO2"."""
    return text.translate(_SUBSCRIPTS)


def names_in(text: str) -> list[str]:
    """
    The names a text states: acronyms and formulas (NADH, CO2, LCA) and capitalised words
    inside a sentence (Rauschenberg). A compound ("Earth-based") is named by its first part.
    """
    names: dict[str, None] = {}
    for sentence in _SENTENCE.split(text):
        for index, word in enumerate(_WORD.findall(sentence)):
            head = plain_digits(word).split("-")[0]
            capitals = sum(char.isupper() for char in head)
            is_acronym = capitals >= 2 or (
                capitals >= 1 and any(c.isdigit() for c in head)
            )
            is_proper_name = (
                index > 0 and len(head) > 1 and head[0].isupper() and head[1].islower()
            )
            if (is_acronym or is_proper_name) and head.lower() not in COMMON_NAMES:
                names[word] = None
    return list(names)


def unbacked_names(
    text: str, resources: list[RoledResource], course: ConfirmedCourse
) -> list[str]:
    """
    The names of a change that neither its direct resources nor the course contain: a
    named example needs a resource that directly supports it, a context one does not count.
    """
    direct = resource_text(
        [r.resource.document for r in resources if r.role == "direct"]
    )
    known = plain_digits(f"{direct} {normalise(course_block(course))}")

    def stem(name: str) -> str:
        wanted = plain_digits(normalise(name))
        if len(wanted) <= MIN_STEM:
            return wanted
        return wanted[: max(MIN_STEM, len(wanted) - 3)]

    return [name for name in names_in(text) if stem(name) not in known]


# ---- the support, capped ----

_RANK: tuple[str, ...] = ("none", "partial", "strong")


def at_most(support: AdaptationSupport, limit: AdaptationSupport) -> AdaptationSupport:
    return support if _RANK.index(support) <= _RANK.index(limit) else limit


def with_checks(
    proposal: AdaptationProposal,
    verification: Verification | None,
    resources: list[RoledResource],
    course: ConfirmedCourse,
) -> AdaptationProposal:
    """
    The proposal with what its check found, its support never raised:
    - a wrong claim, or a change judged forced that no direct resource quotes: none;
    - forced but quoted by a direct resource (the check is a second opinion, not always
      right), a claim not established, a name nothing backs, or no check made: partial.
    """
    names = unbacked_names(proposal.suggested, resources, course)
    is_directly_quoted = any(r.role == "direct" and r.quote for r in proposal.resources)
    limit: AdaptationSupport
    if verification is None:
        limit = "partial"
    elif verification.has_incorrect or (
        verification.is_forced and not is_directly_quoted
    ):
        limit = "none"
    elif verification.is_forced or verification.has_unestablished or names:
        limit = "partial"
    else:
        limit = "strong"
    return proposal.model_copy(
        update={
            "support": at_most(proposal.support, limit),
            "checks": AdaptationChecks(
                status="checked" if verification else "unchecked",
                problems=verification.problems if verification else [],
                unbacked_names=names,
            ),
        }
    )


# ---- the loop ----


async def propose(
    course: ConfirmedCourse,
    connection: ChosenConnection,
    resources: list[RoledResource],
    lang: TutorLang,
    problems: list[str] | None = None,
) -> AdaptationProposal:
    """One proposal. Raises UnusableLLMOutputError when none can be used after a retry."""
    data: dict = {
        "course": course_block(course),
        "connection": connection_block(connection),
    }
    if resources:
        data["resource"] = resource_blocks(resources)
    if problems:
        data["problems"] = problems_block(problems)

    def usable(answer: AdaptationAnswer) -> bool:
        return to_proposal(answer, course, connection, resources) is not None

    answer = await structured_call(
        "adaptation",
        lang,
        AdaptationAnswer,
        data,
        {
            "count": len(resources),
            "greencomp": greencomp_for_prompt(),
            "has_problems": bool(problems),
        },
        is_usable=usable,
    )
    return to_proposal(answer, course, connection, resources)  # type: ignore[return-value]


async def check(
    course: ConfirmedCourse,
    proposal: AdaptationProposal,
    resources: list[RoledResource],
    lang: TutorLang,
) -> Verification | None:
    """The check of a proposal's claims; None when it could not be made."""
    change = f"{proposal.suggested} {proposal.why}"
    try:
        answer = await structured_call(
            "verification",
            lang,
            VerificationAnswer,
            {
                "course": course_block(course),
                "proposal": f"Proposed change: {proposal.suggested}\nWhy: {proposal.why}",
                "resource": resource_blocks(resources),
            },
            {"count": len(resources)},
        )
    except LLM_FAILURES:
        logger.warning("tutor_adaptation_unchecked")
        return None
    return to_verification(answer, resources, change)


async def adapt_course(
    course: ConfirmedCourse, connection: ChosenConnection, lang: TutorLang
) -> AdaptationProposal:
    """
    One checked change for the connection. Raises UnusableLLMOutputError when no proposal
    can be used, LLMUnavailableError when the model does not answer it. A check or a
    second proposal that fails leaves the first proposal, unchecked or as it was.

    A change that cites no resource (the connection has none, or the model used none) has
    nothing to be checked against: its support is already "none", which no check can
    lower, so it comes back without a check (`checks` null).

    Traced as one run in LangSmith, its LLM calls inside, and summed up in one log line
    without content (`tutor_adaptation_done`), to measure how often each step runs.
    """
    resources = roled_resources(connection)
    proposal = await propose(course, connection, resources, lang)
    calls = 1
    if not proposal.resources:
        _log_adaptation(calls, proposal)
        return proposal

    verification = await check(course, proposal, resources, lang)
    calls += 1
    first_problems = len(verification.problems) if verification else None
    # would a wrong claim or a forced change alone have asked again? (for the log)
    first_serious = (
        verification.has_incorrect or verification.is_forced if verification else None
    )
    is_redone = is_second_kept = False
    if verification and verification.problems:
        # asked again once without what was wrong; kept only once checked too
        is_redone = True
        try:
            second = await propose(
                course, connection, resources, lang, verification.problems
            )
        except LLM_FAILURES:
            second = None
        calls += 1
        second_check = None
        if second:
            second_check = await check(course, second, resources, lang)
            calls += 1
        if second and second_check:
            proposal, verification = second, second_check
            is_second_kept = True
    proposal = with_checks(proposal, verification, resources, course)
    _log_adaptation(
        calls, proposal, first_problems, first_serious, is_redone, is_second_kept
    )
    return proposal


def _log_adaptation(
    calls: int,
    proposal: AdaptationProposal,
    first_problems: int | None = None,
    first_serious: bool | None = None,
    is_redone: bool = False,
    is_second_kept: bool = False,
) -> None:
    """
    One line per adaptation, counts and enums only (never the course or the change).
    `calls` counts the steps that asked the model, not an unusable answer asked again.
    """
    checks = proposal.checks
    logger.info(
        "tutor_adaptation_done calls=%s check=%s first_problems=%s first_serious=%s "
        "redone=%s second_kept=%s final_problems=%s unbacked_names=%s support=%s kind=%s",
        calls,
        checks.status if checks else "none",
        first_problems,
        first_serious,
        is_redone,
        is_second_kept,
        len(checks.problems) if checks else None,
        len(checks.unbacked_names) if checks else None,
        proposal.support,
        proposal.kind,
    )
