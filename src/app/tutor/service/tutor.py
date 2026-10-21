from langchain_core.runnables import RunnableConfig
from langchain_mistralai import ChatMistralAI  # type: ignore

from src.app.core.config import Settings
from src.app.shared.utils.utils import extract_doc_info
from src.app.tutor.service.agents import (
    FeedbackAgent,
    PedagogicalEngineerAgent,
    SDGExpertAgent,
    UniversityTeacherAgent,
    get_disciplinary_skills,
)
from src.app.tutor.service.models import (
    MessageWithResources,
    SyllabusFeedback,
    SyllabusResponseAgent,
    TutorSyllabusRequest,
)
from src.app.tutor.service.syllabus import (
    GREENCOMP_COMPETENCIES,
    detect_syllabus_lang,
    generate_syllabus,
    render_references,
    split_references,
    trim_chatter,
)

chat_model: ChatMistralAI | None = None


async def init_chat_model(settings) -> None:
    global chat_model
    if chat_model is None:
        chat_model = ChatMistralAI(
            model_name=settings.MISTRAL_LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
        )


async def close_chat_model() -> None:
    global chat_model
    if chat_model is not None:
        await chat_model.aclose()
        chat_model = None


async def tutor_manager(
    content: TutorSyllabusRequest,
    lang: str,
    settings: Settings,
    trace_context: dict | None = None,
) -> list[SyllabusResponseAgent]:
    formatted_content = MessageWithResources(
        lang=lang,
        content=content.extracts,
        resources=extract_doc_info(content.documents),
        themes=[theme for extract in content.extracts for theme in extract.themes],
        summary=[extract.summary for extract in content.extracts],
        course_title=content.course_title,
        discipline=content.discipline,
        level=content.level,
        duration=content.duration,
        description=content.description,
    )

    if chat_model is None:
        raise RuntimeError(
            "Chat model not initialized. Call init_chat_model() at startup."
        )

    base_tags = ["welearn", "tutor", "syllabus"]
    base_metadata = {
        "component": "tutor_syllabus",
        "environment": settings.ENV,
        "language": lang,
        "course_title": content.course_title,
    }

    if trace_context:
        base_metadata.update(trace_context)
        endpoint = trace_context.get("endpoint")
        if endpoint:
            base_tags.append(f"endpoint:{endpoint}")

    if settings.TUTOR_SINGLE_PASS:
        content_md = await generate_syllabus(
            formatted_content,
            chat_model,
            get_disciplinary_skills().get(formatted_content.discipline, []),
            RunnableConfig(
                tags=base_tags + ["agent:single_pass"],
                metadata={**base_metadata, "agent": "SinglePassSyllabus"},
                run_name="Tutor (SinglePassSyllabus)",
            ),
        )
        # source name kept: the client picks the "PedagogicalEngineerAgent" item
        return [
            SyllabusResponseAgent(content=content_md, source="PedagogicalEngineerAgent")
        ]

    teacher_agent = UniversityTeacherAgent(
        chat_model,
        lang,
        trace_tags=base_tags,
        trace_metadata=base_metadata,
    )
    sdg_agent = SDGExpertAgent(
        chat_model,
        GREENCOMP_COMPETENCIES,
        lang,
        trace_tags=base_tags,
        trace_metadata=base_metadata,
    )
    pedagogical_agent = PedagogicalEngineerAgent(
        chat_model,
        GREENCOMP_COMPETENCIES,
        lang,
        trace_tags=base_tags,
        trace_metadata=base_metadata,
    )

    teacher_response = await teacher_agent.generate(formatted_content)
    sdg_response = await sdg_agent.enhance(
        teacher_response, formatted_content.resources, lang
    )
    pedagogical_response = await pedagogical_agent.refine(sdg_response)
    # references are never trusted from the LLM: rebuilt from the selected documents
    body, _ = split_references(pedagogical_response.content)
    pedagogical_response.content = (
        trim_chatter(body)
        + "\n\n"
        + render_references(formatted_content.resources, lang)
    )

    return [teacher_response, sdg_response, pedagogical_response]


async def apply_feedback(
    body: SyllabusFeedback, settings: Settings, trace_context: dict | None = None
) -> str:
    """Apply the teacher's feedback to the syllabus body; references are kept verbatim."""
    if chat_model is None:
        raise RuntimeError(
            "Chat model not initialized. Call init_chat_model() at startup."
        )
    syllabus_body, references = split_references(body.syllabus[0].content)
    # same tags/metadata as generation; the client sends no lang here, so it is
    # read from the syllabus's own headings
    tags = ["welearn", "tutor", "syllabus"]
    if trace_context and trace_context.get("endpoint"):
        tags.append(f"endpoint:{trace_context['endpoint']}")
    agent = FeedbackAgent(
        chat_model,
        GREENCOMP_COMPETENCIES,
        trace_tags=tags,
        trace_metadata={
            "component": "tutor_syllabus",
            "environment": settings.ENV,
            "language": detect_syllabus_lang(syllabus_body),
            **(trace_context or {}),
        },
    )
    new_body = trim_chatter(
        await agent.apply(syllabus_body, body.feedback, body.extracts)
    )
    return f"{new_body}\n\n{references}" if references else new_body
