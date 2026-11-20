"""
Syllabus v2: sustainability connections for a confirmed course
(`POST /tutor/course/connection_ideas`).

The `ideas` prompt reads the course and suggests a few connections, each built on one part
of the course. The part it names is then matched to the course as the teacher wrote it, so
the client can show and place it; a part that is not in the course is dropped.
"""

from src.app.tutor.domain.connections import (
    MAX_IDEAS,
    MAX_QUERY_CHARS,
    ConfirmedCourse,
    ConnectionIdea,
    ConnectionIdeasResponse,
    IdeasAnswer,
)
from src.app.tutor.domain.models import TutorLang
from src.app.tutor.services.course_reading import normalise
from src.app.tutor.services.llm import structured_call

MAX_QUERIES = 2
# a label names a course part in a few words
MAX_LABEL_WORDS = 8


def course_block(course: ConfirmedCourse) -> str:
    """The course as the prompts read it: one labelled line or list per field."""

    def line(label: str, value: str | None) -> str:
        return f"{label}: {value}" if value else ""

    def bullets(label: str, items: list[str]) -> str:
        return f"{label}:\n" + "\n".join(f"- {item}" for item in items) if items else ""

    topics = [
        f"{t.title} ({', '.join(t.subtopics)})" if t.subtopics else t.title
        for t in course.topics
    ]
    return "\n".join(
        part
        for part in [
            line("Title", course.title),
            line("Discipline", course.discipline),
            line("Level", course.level),
            line("Overview", course.overview),
            bullets("Topics", topics),
            bullets("Learning objectives", course.objectives)
            or "Learning objectives: none stated in the syllabus",
            bullets("Competencies", course.competencies),
            bullets("Course outline", course.sessions),
        ]
        if part
    )


def course_parts(course: ConfirmedCourse) -> list[str]:
    """The parts of the course a connection may build on, each as the teacher wrote it."""
    return [
        *(part for t in course.topics for part in [t.title, *t.subtopics]),
        *course.objectives,
        *course.competencies,
        *course.sessions,
    ]


def part_of_course(anchor: str | None, parts: list[str]) -> str | None:
    """
    The course part the model names, as the teacher wrote it: the part that contains it,
    or else the longest part it contains. None when it is not in the course.
    """
    if not anchor:
        return None
    wanted = normalise(anchor)
    known = [(part, normalise(part)) for part in parts]
    known = [(part, text) for part, text in known if text]
    containing = next((part for part, text in known if wanted in text), None)
    if containing:
        return containing
    contained = sorted(
        ((part, text) for part, text in known if text in wanted),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    return contained[0][0] if contained else None


def to_ideas(answer: IdeasAnswer, parts: list[str]) -> ConnectionIdeasResponse:
    """The model's ideas tied to the course: numbered, at most MAX_IDEAS."""
    ideas = []
    for n, idea in enumerate(answer.ideas[:MAX_IDEAS], start=1):
        anchor = part_of_course(idea.anchor, parts)
        label = idea.anchor_label
        ideas.append(
            ConnectionIdea(
                id=f"idea-{n}",
                title=idea.title,  # type: ignore[arg-type]
                anchor=anchor,
                # a label only names a part that is in the course, and stays short
                anchor_label=(
                    label
                    if anchor and label and len(label.split()) <= MAX_LABEL_WORDS
                    else None
                ),
                why=idea.why,  # type: ignore[arg-type]
                use=idea.use,
                queries=[
                    query[:MAX_QUERY_CHARS] for query in idea.queries[:MAX_QUERIES]
                ],
            )
        )
    return ConnectionIdeasResponse(
        ideas=ideas, none_reason=None if ideas else answer.none_reason
    )


async def find_connection_ideas(
    course: ConfirmedCourse, lang: TutorLang
) -> ConnectionIdeasResponse:
    """
    At most MAX_IDEAS connections for the course, or none with the reason why.
    Raises UnusableLLMOutputError when the model gives neither, even after a retry.
    """
    answer = await structured_call(
        "ideas",
        lang,
        IdeasAnswer,
        {"course": course_block(course)},
        {"max_ideas": MAX_IDEAS},
    )
    return to_ideas(answer, course_parts(course))
