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
import unicodedata

from src.app.shared.domain.exceptions import (
    UnusableLLMOutputError,
    UnusableReadingError,
)
from src.app.shared.utils.logger import logger as utils_logger
from src.app.tutor.domain.course import (
    CourseOutline,
    CourseReading,
    CourseReadResponse,
    ReadObjective,
    ReadSession,
    ReadTopic,
)
from src.app.tutor.domain.models import TutorLang
from src.app.tutor.services.llm import MAX_DATA_CHARS, structured_call

logger = utils_logger(__name__)

# a focus is a course title picked among courses_found
MAX_FOCUS_CHARS = 200


# ---- comparing texts (port of courseCheck.ts) ----


def normalise(text: str) -> str:
    """
    A text as it reads: case, spaces, line breaks, quote and dash styles aside. Accents
    and words are kept.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"[«»“”]", '"', text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.replace("…", "...").replace("œ", "oe")
    # a word cut at the end of a line ("struc-\nture") is one word
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", "", text)


_LIST_MARKER = re.compile(r"^\s*(?:[-–•o▪◦*]|\d+[.)])\s")


def core(text: str) -> str:
    """normalise, without a list marker or a final punctuation mark."""
    return re.sub(r"[.;:,!]+$", "", normalise(_LIST_MARKER.sub("", text, count=1)))


# ---- the parts of a long document ----


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
    mine, theirs = core(numbered.title), core(topic.source or topic.title)
    return bool(mine) and bool(theirs) and (mine in theirs or theirs in mine)


def _is_heading_line(title: str, text: str) -> bool:
    """A line of its own in the syllabus, not words inside a sentence."""
    wanted = core(title)
    return bool(wanted) and any(
        wanted in core(line) and len(core(line)) <= len(wanted) + MAX_LABEL
        for line in text.splitlines()
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

    no_duplicates = len(unique) == len(topics)
    is_faithful = (
        no_duplicates
        and len(unique) == len(numbered)
        and all(_is_same_topic(unique[i], item) for i, item in enumerate(numbered))
    )
    # numbered items kept under the syllabus's own higher headings ("Partie II : …" with its
    # chapters as subtopics): a faithful structure too
    is_under_own_headings = (
        no_duplicates
        and all(
            any(
                _is_same_topic(topic, item)
                or any(
                    _is_same_topic(ReadTopic(title=sub, source=sub), item)
                    for sub in topic.subtopics
                )
                for topic in unique
            )
            for item in numbered
        )
        and all(_is_heading_line(topic.title, text) for topic in unique)
    )
    if is_under_own_headings:
        return unique

    if is_faithful:
        # one for one, but a title may be a category from the prose: the heading wins
        result = []
        for topic, heading in zip(unique, numbered):
            mine, theirs = core(heading.title), core(topic.title)
            is_own_words = theirs in mine or mine in theirs
            result.append(
                topic
                if is_own_words
                else heading.model_copy(
                    update={"subtopics": _subtopics_of(topic, numbered)}
                )
            )
        return result

    result = []
    for item in numbered:
        same = next((t for t in unique if _is_same_topic(t, item)), None)
        subtopics = _subtopics_of(same, numbered) if same else []
        result.append(item.model_copy(update={"subtopics": subtopics}))
    return result


# ---- cleaning the model's answer ----


def _text(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _texts(values: list[str]) -> list[str]:
    return [text for text in (_text(v) for v in values) if text]


def clean_topics(topics: list[ReadTopic]) -> list[ReadTopic]:
    """Topics with a title, their values trimmed."""
    return [
        ReadTopic(
            title=title, subtopics=_texts(topic.subtopics), source=_text(topic.source)
        )
        for topic in topics
        if (title := _text(topic.title))
    ]


def clean_reading(reading: CourseReading) -> CourseReading:
    """Empty values dropped, and a title that is not a course's name removed."""
    title = _text(reading.title)
    discipline = _text(reading.discipline)
    return reading.model_copy(
        update={
            "title": title if is_course_title(title) else None,
            "discipline": discipline,
            # inferred by the model: kept only when no discipline is stated
            "discipline_suggested": (
                None if discipline else _text(reading.discipline_suggested)
            ),
            "level": _text(reading.level),
            "document_language": _text(reading.document_language),
            "description": _text(reading.description),
            "summary": _text(reading.summary),
            "courses_found": _texts(reading.courses_found),
            "objectives": [
                ReadObjective(text=text, scope=objective.scope)
                for objective in reading.objectives
                if (text := _text(objective.text))
            ],
            "topics": clean_topics(reading.topics),
            # a session with only its label ("Week 8") is still a session
            "sessions": [
                ReadSession(label=_text(s.label), title=_text(s.title))
                for s in reading.sessions
                if _text(s.label) or _text(s.title)
            ],
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
    be read for its outline is skipped, and the response says it is partial.
    """
    parts = text_parts(text)
    focus = _text(focus)
    focus_data = {"focus": focus[:MAX_FOCUS_CHARS]} if focus else {}
    variables = {"has_focus": bool(focus)}

    try:
        reading = await structured_call(
            "read", lang, CourseReading, {"course": parts[0], **focus_data}, variables
        )
    except UnusableLLMOutputError:
        raise UnusableReadingError()
    reading = clean_reading(reading)

    # the first part was read in full: the outline only reads the rest, unless the
    # reading found no topics
    topics = list(reading.topics)
    summary = reading.summary
    is_partial = False
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
        summary = summary or _text(outline.summary)
        known = {normalise(topic.title) for topic in topics}
        for topic in clean_topics(outline.topics):
            if normalise(topic.title) not in known:
                topics.append(topic)
                known.add(normalise(topic.title))

    if is_partial:
        logger.warning("tutor_course_read_partial parts=%s", len(parts))
    return CourseReadResponse(
        text=text,
        is_text_truncated=len(parts) > 1,
        is_partial=is_partial,
        reading=reading.model_copy(
            update={
                "summary": summary,
                "topics": prefer_numbered_topics(topics, text),
            }
        ),
    )
