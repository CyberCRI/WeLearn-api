"""
Syllabus v2: what the model reads of the course, a connection and the WeLearn resources,
the same for every prompt.
"""

from src.app.shared.domain.documents import EnhancedDocumentSlice
from src.app.tutor.domain.connections import MAX_RESOURCE_CHARS, ConfirmedCourse
from src.app.tutor.services.llm import DataBlock


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


def connection_text(
    title: str, anchor: str | None, why: str, use: str | None = None
) -> str:
    """A connection as the prompts read it: its title, the part of the course, why."""
    lines = [
        f"Title: {title}",
        f"Part of the course concerned: {anchor or 'the whole course'}",
        f"Why: {why}",
    ]
    if use:
        lines.append(f"Use of the course's concept: {use}")
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


def resource_block(doc: EnhancedDocumentSlice, role: str | None = None) -> DataBlock:
    """A resource as the prompts read it: its passage, cut to MAX_RESOURCE_CHARS, with its
    role (when it has one), title and source as attributes."""
    attrs = {"role": role} if role else {}
    return DataBlock(
        text=(doc.payload.slice_content or "")[:MAX_RESOURCE_CHARS],
        attrs={**attrs, "title": doc.payload.document_title, "source": source_of(doc)},
    )


def nth(items: list, n: int | None):
    """The item the model named by its number (from 1); None for a number out of range."""
    return items[n - 1] if n is not None and 1 <= n <= len(items) else None
