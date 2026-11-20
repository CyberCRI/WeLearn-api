"""
Syllabus v2: read the course document a teacher uploads (`POST /tutor/course/read`).

The `read` prompt reads the first part of the document (MAX_DATA_CHARS). A longer document
is completed part by part by `read_outline` (summary and topics only), and the topics found
twice are merged. The answer is then checked against the text, as the prototype did on the
client (`courseReading.ts`): a title that is a person or a date is dropped, and when the
syllabus numbers its themes ("Theme 1: …"), its own headings win over the model's regrouping.
The word-for-word check of each value stays on the client, which gets the text back.
"""

import re

from src.app.shared.domain.exceptions import (
    LLMUnavailableError,
    UnusableLLMOutputError,
    UnusableReadingError,
)
from src.app.shared.utils.logger import logger as utils_logger
from src.app.tutor.domain.course import (
    CourseOutline,
    CourseReading,
    CourseReadResponse,
    ReadTopic,
)
from src.app.tutor.domain.models import TutorLang
from src.app.tutor.services.llm import MAX_DATA_CHARS, structured_call
from src.app.tutor.services.text_match import core, names_each_other, normalise

logger = utils_logger(__name__)

# a focus is a course title picked among courses_found
MAX_FOCUS_CHARS = 200


# ---- the parts of a long document ----


# each part is one LLM call, made in turn: a longer document is read in its first parts only
MAX_PARTS = 4


def text_parts(text: str, size: int = MAX_DATA_CHARS) -> list[str]:
    """Consecutive parts of `size` characters; a short text is one part."""
    return [text[start : start + size] for start in range(0, len(text), size)] or [""]


# ---- the course title ----

# a line about a person, a session or a date, or the document's own label
_NOT_A_TITLE = [
    re.compile(
        r"^\s*(professeur\w*|professor|prof\.?|enseignant\w*|instructor|lecturer|teacher"
        r"|tutor|responsable\w*|coordinat\w*|charg[ée]e? de cours|intervenant\w*|contact"
        r"|e-?mail|office hours)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(session|semester|semestre|term|year|ann[ée]e|date|period|p[ée]riode)\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(syllabus|course syllabus|course outline|plan de cours|programme|program"
        r"|fiche (ue|cours|de cours))\s*$",
        re.IGNORECASE,
    ),
]
_TWO_LETTERS = re.compile(r"[^\W\d_]{2}")


def is_course_title(value: str | None) -> bool:
    """False for a teacher line, a date, a bare number or a label such as "Syllabus"."""
    return (
        bool(value)
        and bool(_TWO_LETTERS.search(value))  # type: ignore[arg-type]
        and not any(p.search(value) for p in _NOT_A_TITLE)  # type: ignore[arg-type]
    )


# ---- numbered topics (port of courseReading.ts) ----

# "Theme 3: Freedom of expression…", "Chapitre 2 – …"; weeks and sessions are not topics
_NUMBERED = re.compile(
    r"^\s*(theme|th[èe]me|topic|chapter|chapitre|part|partie|module|unit|unit[ée]|axe)"
    r"\s*(\d{1,2})\s*[:.–—-]\s*(\S.*)$",
    re.IGNORECASE,
)
# a heading cut at the end of a line (after a comma or a small word) goes on to the next
_GOES_ON = re.compile(
    r"(,|-|–|\b(and|or|of|the|for|to|in|on|with|a|an|et|ou|de|des|du|la|le|les|à|au|aux"
    r"|en|pour|sur|par|dans|avec|d['’]|l['’]))\s*$",
    re.IGNORECASE,
)
MIN_NUMBERED = 3
# a heading line may carry a label before it ("Partie II :")
MAX_LABEL = 20


def numbered_topics(text: str) -> list[ReadTopic]:
    """The syllabus's own numbered topics: the first run of one label numbered 1, 2, 3…"""
    lines = text.splitlines()
    items: list[tuple[str, int, str]] = []
    i = 0
    while i < len(lines):
        match = _NUMBERED.match(lines[i])
        if match:
            title = match.group(3).strip()
            while (
                _GOES_ON.search(title)
                and i + 1 < len(lines)
                and lines[i + 1].strip()
                and not _NUMBERED.match(lines[i + 1])
            ):
                i += 1
                title = f"{title} {lines[i].strip()}"
            items.append(
                (match.group(1).lower(), int(match.group(2)), title.rstrip(" \t,;:"))
            )
        i += 1

    for label in dict.fromkeys(label for label, _, _ in items):
        run: list[str] = []
        for item_label, n, title in items:
            if item_label == label and n == len(run) + 1:
                run.append(title)
        if len(run) >= MIN_NUMBERED:
            return [ReadTopic(title=title, source=title) for title in run]
    return []


def merge_duplicate_topics(topics: list[ReadTopic]) -> list[ReadTopic]:
    """The same topic named twice: one topic, its subtopics together."""
    merged: list[ReadTopic] = []
    for topic in topics:
        same = next((t for t in merged if core(t.title) == core(topic.title)), None)
        if same is None:
            merged.append(topic.model_copy(update={"subtopics": list(topic.subtopics)}))
            continue
        for sub in topic.subtopics:
            if not any(core(known) == core(sub) for known in same.subtopics):
                same.subtopics.append(sub)
    return merged


def _is_same_topic(topic: ReadTopic, numbered: ReadTopic) -> bool:
    return names_each_other(numbered.title, topic.source or topic.title, norm=core)


def _is_heading_line(title: str, line_cores: list[str]) -> bool:
    """A line of its own in the syllabus (given by its lines' `core`), not words inside a
    sentence."""
    wanted = core(title)
    return bool(wanted) and any(
        wanted in line and len(line) <= len(wanted) + MAX_LABEL for line in line_cores
    )


def _subtopics_of(topic: ReadTopic, numbered: list[ReadTopic]) -> list[str]:
    """Without the subtopics that are a numbered heading (its own, or another theme's)."""
    return [
        sub
        for sub in topic.subtopics
        if not any(core(sub) in core(other.title) for other in numbered)
    ]


def prefer_numbered_topics(topics: list[ReadTopic], text: str) -> list[ReadTopic]:
    """
    The model's topics checked against the syllabus's numbered topics. When the syllabus
    numbers them and the model's list does not follow them one for one (grouped under
    categories from the prose, repeated, renamed), the syllabus's headings are kept, in its
    order, with the subtopics the model gave for the same heading. Otherwise the model's
    topics, without duplicates.
    """
    unique = merge_duplicate_topics(topics)
    numbered = numbered_topics(text) if text else []
    if not numbered:
        return unique

    has_duplicates = len(unique) != len(topics)
    if not has_duplicates and _is_under_own_headings(unique, numbered, text):
        return unique
    if not has_duplicates and _follows_one_for_one(unique, numbered):
        return _with_headings_for_categories(unique, numbered)
    return _rebuilt_from_headings(unique, numbered)


def _is_under_own_headings(
    topics: list[ReadTopic], numbered: list[ReadTopic], text: str
) -> bool:
    """
    Numbered items kept under the syllabus's own higher headings ("Partie II : …" with its
    chapters as subtopics): every numbered item is a topic or a subtopic, and every topic
    is a heading line of the syllabus. A faithful structure too.
    """

    def is_kept(item: ReadTopic) -> bool:
        return any(
            _is_same_topic(topic, item)
            or any(
                _is_same_topic(ReadTopic(title=sub, source=sub), item)
                for sub in topic.subtopics
            )
            for topic in topics
        )

    if not all(is_kept(item) for item in numbered):
        return False
    # each line of the syllabus normalised once, not once per topic
    line_cores = [core(line) for line in text.splitlines()]
    return all(_is_heading_line(topic.title, line_cores) for topic in topics)


def _follows_one_for_one(topics: list[ReadTopic], numbered: list[ReadTopic]) -> bool:
    """The model's topics are the numbered ones, in the same order."""
    return len(topics) == len(numbered) and all(
        _is_same_topic(topic, item) for topic, item in zip(topics, numbered)
    )


def _with_headings_for_categories(
    topics: list[ReadTopic], numbered: list[ReadTopic]
) -> list[ReadTopic]:
    """One for one, but a title may be a category from the prose: the heading wins."""
    return [
        (
            topic
            if _names_the_heading(topic, heading)
            else heading.model_copy(
                update={"subtopics": _subtopics_of(topic, numbered)}
            )
        )
        for topic, heading in zip(topics, numbered)
    ]


def _names_the_heading(topic: ReadTopic, heading: ReadTopic) -> bool:
    mine, theirs = core(heading.title), core(topic.title)
    return theirs in mine or mine in theirs


def _rebuilt_from_headings(
    topics: list[ReadTopic], numbered: list[ReadTopic]
) -> list[ReadTopic]:
    """The syllabus's headings, in its order, with the subtopics the model gave for each."""
    result = []
    for item in numbered:
        same = next((t for t in topics if _is_same_topic(t, item)), None)
        subtopics = _subtopics_of(same, numbered) if same else []
        result.append(item.model_copy(update={"subtopics": subtopics}))
    return result


# ---- what the text alone does not settle ----
# (texts are trimmed and empty values dropped as the answer is parsed: domain/course.py)


def clean_reading(reading: CourseReading) -> CourseReading:
    """A title that is not a course's name removed, a discipline stated beats a suggested one."""
    return reading.model_copy(
        update={
            "title": reading.title if is_course_title(reading.title) else None,
            # inferred by the model: kept only when no discipline is stated
            "discipline_suggested": (
                None if reading.discipline else reading.discipline_suggested
            ),
        }
    )


# ---- the reading ----


async def read_course(
    text: str, lang: TutorLang, focus: str | None = None
) -> CourseReadResponse:
    """
    Read a course document's text. `focus` reads only one of the courses the document
    holds (a title from a first reading's `courses_found`).

    Raises UnusableReadingError when the first part can't be read. A later part that can't
    be read for its outline is skipped, and the response says it is partial; so are the
    parts beyond MAX_PARTS, and the rest once the model stops answering.
    """
    every_part = text_parts(text)
    parts = every_part[:MAX_PARTS]
    focus = focus.strip() if focus else None
    focus_data = {"focus": focus} if focus else {}
    variables = {"has_focus": bool(focus)}

    try:
        reading = await structured_call(
            "read", lang, CourseReading, {"course": parts[0], **focus_data}, variables
        )
    except UnusableLLMOutputError as e:
        raise UnusableReadingError() from e
    reading = clean_reading(reading)

    # the first part was read in full: the outline only reads the rest, unless the
    # reading found no topics
    topics = list(reading.topics)
    summary = reading.summary
    is_partial = len(every_part) > MAX_PARTS
    for part in parts[1:] if topics else parts:
        try:
            outline = await structured_call(
                "read_outline",
                lang,
                CourseOutline,
                {"course_part": part, **focus_data},
                variables,
            )
        except UnusableLLMOutputError:
            is_partial = True
            continue
        except LLMUnavailableError:
            # the next parts would each wait out the same timeout
            is_partial = True
            break
        summary = summary or outline.summary
        known = {normalise(topic.title) for topic in topics}
        for topic in outline.topics:
            if normalise(topic.title) not in known:
                topics.append(topic)
                known.add(normalise(topic.title))

    if is_partial:
        logger.warning("tutor_course_read_partial parts=%s", len(every_part))
    return CourseReadResponse(
        text=text,
        is_text_truncated=len(every_part) > 1,
        is_partial=is_partial,
        reading=reading.model_copy(
            update={
                "summary": summary,
                "topics": prefer_numbered_topics(topics, text),
            }
        ),
    )
