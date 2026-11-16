from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from src.app.baml_client.async_client import b, types
from src.app.core.config import Settings
from src.app.data_collection.services.data_collection import get_data_collection_service
from src.app.search.domain.models import EnhancedSearchQuery
from src.app.search.services.search import SearchService, get_search_service
from src.app.search.services.search_helpers import search_multi_inputs
from src.app.shared.domain.exceptions import NoResultsError
from src.app.shared.domain.responses import ErrorResponse, MessageResponse
from src.app.shared.utils.db_errors import describe_error
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as utils_logger
from src.app.shared.utils.requests import extract_session_cookie
from src.app.shared.utils.utils import check_upload_limits, get_files_content
from src.app.tutor.domain.course import CourseReadResponse
from src.app.tutor.domain.models import (
    CompetencyMappingRequest,
    CourseDescriptionRequest,
    ExtractorOutputList,
    IntegrateSustainabilityRequest,
    LearningObjectivesRequest,
    LearningOutcomesRequest,
    MasFileLang,
    SummariesList,
    SyllabusFeedback,
    SyllabusGenerationRequest,
    SyllabusGenerationResponse,
    SyllabusResponse,
    SyllabusResponseAgent,
    SyllabusUserUpdate,
    TutorLang,
    TutorSearchResponse,
    TutorSyllabusRequest,
    UserInput,
)
from src.app.tutor.services.course_reading import MAX_FOCUS_CHARS, read_course
from src.app.tutor.services.extractor import extract_files_summaries, truncate_documents
from src.app.tutor.services.orchestrator import SyllabusOrchestrator
from src.app.tutor.services.tutor import apply_feedback, tutor_manager
from src.app.user.services.guest_quota import (
    GUEST_QUOTA_RESPONSES,
    GuestQuota,
    guest_quota,
)

logger = utils_logger(__name__)

router = APIRouter()

# upload limits of the file extraction routes; the client allows 3 files of 5 MB
MAX_UPLOAD_FILES = 5
MAX_UPLOAD_FILE_SIZE = 10 * 1024 * 1024
DOCUMENT_SEPARATOR = "__DOCUMENT_SEPARATOR__"

UPLOAD_ERROR_RESPONSES: dict[int | str, dict] = {
    400: {
        "model": ErrorResponse,
        "description": "Too many files (code TOO_MANY_FILES) or unreadable file",
    },
    413: {
        "model": ErrorResponse,
        "description": "A file is too large (code FILE_TOO_LARGE)",
    },
}


async def _read_documents(files: list[UploadFile]) -> str:
    """Check the upload limits, extract each file's text and fit it to the prompt."""
    await check_upload_limits(files, MAX_UPLOAD_FILES, MAX_UPLOAD_FILE_SIZE)
    files_content = await get_files_content(files)
    return DOCUMENT_SEPARATOR.join(truncate_documents(files_content))


# syllabus v2: one course document, 5 MB (as the client allows)
MAX_COURSE_FILE_SIZE = 5 * 1024 * 1024

COURSE_READ_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "Unsupported or unreadable file"},
    413: {
        "model": ErrorResponse,
        "description": "The file is too large (code FILE_TOO_LARGE)",
    },
    422: {
        "model": ErrorResponse,
        "description": (
            "No text in the file, e.g. a scanned PDF (code EMPTY_DOCUMENT_TEXT), "
            "or invalid form fields"
        ),
    },
    502: {
        "model": ErrorResponse,
        "description": "The document could not be read (code UNUSABLE_READING)",
    },
}


@router.post(
    "/course/read",
    response_model=CourseReadResponse,
    responses={**COURSE_READ_RESPONSES, **GUEST_QUOTA_RESPONSES},
    summary="Read a course document (syllabus v2)",
)
async def read_course_document(
    file: Annotated[UploadFile, File()],
    lang: Annotated[TutorLang, Form()] = "en",
    focus: Annotated[str | None, Form(max_length=MAX_FOCUS_CHARS)] = None,
    quota: GuestQuota = Depends(guest_quota("syllabus")),
) -> CourseReadResponse:
    """
    The course in the uploaded document: title, objectives, topics, sessions... and the
    extracted text, so the client checks each value word for word. `lang` is the teacher's
    language (the answer's); `focus` reads one course of a document that holds several.

    A guest's reading counts as one of their daily syllabi, refunded if it fails. A
    file rejected before the reading (size, type, no text) costs nothing.
    """
    await check_upload_limits([file], 1, MAX_COURSE_FILE_SIZE)
    [text] = await get_files_content([file])
    with quota.spend():
        return await read_course(text, lang, focus)


@router.post("/files/content", responses=UPLOAD_ERROR_RESPONSES)
async def extract_files_content(
    files: Annotated[list[UploadFile], File()],
    response: Response,
    lang: TutorLang = "en",
) -> ExtractorOutputList | None:
    files_content_str = await _read_documents(files)

    try:
        return await extract_files_summaries(files_content_str, lang)

    except Exception as e:
        logger.error(f"Error in extractor schema: {e}")
        response.status_code = 204
        raise e


@router.post("/mas_files/content", responses=UPLOAD_ERROR_RESPONSES)
async def extract_files_content_baml(
    files: Annotated[list[UploadFile], File()],
    response: Response,
    lang: MasFileLang = "en",
    mode: str = "mode_1",
) -> types.FileSummary | None:
    files_content_str = await _read_documents(files)

    mode_map = {"mode_1": "article", "mode_2": "syllabus", "mode_3": None}

    try:
        result = await b.FileSummaryExtractor(
            mode=mode_map[mode],  # or "article", "syllabus", etc.
            file_content=files_content_str,
            output_language=lang,
        )

        return result

    except Exception as e:
        logger.error(f"Error in extractor schema: {e}")
        response.status_code = 204
        raise e


@router.post("/search_extracts", response_model=TutorSearchResponse)
async def tutor_search_extract(
    summaries: SummariesList,
    background_tasks: BackgroundTasks,
    response: Response,
    sp: SearchService = Depends(get_search_service),
    nb_results: int = Query(15, ge=1, le=100),
):
    try:
        qp = EnhancedSearchQuery(
            query=summaries.summaries,
            nb_results=nb_results,
            sdg_filter=None,
            corpora=None,
        )

        search_results = await search_multi_inputs(
            qp=qp,
            background_tasks=background_tasks,
            callback_function=sp.search_handler,
        )

        if search_results:
            search_results = [
                point.model_copy(update={"vector": None}) for point in search_results
            ]

    except NoResultsError as e:
        response.status_code = 404
        logger.error(f"No results found: {e}")
        return TutorSearchResponse(
            extracts=[],
            nb_results=0,
            documents=[],
        )

    if not search_results:
        return TutorSearchResponse(
            extracts=[],
            nb_results=0,
            documents=[],
        )

    resp = TutorSearchResponse(
        extracts=[],
        nb_results=len(search_results),
        documents=search_results,
    )

    return resp


@router.post("/syllabus", responses=GUEST_QUOTA_RESPONSES)
async def create_syllabus(
    request: Request,
    body: TutorSyllabusRequest,
    lang: TutorLang = "en",
    data_collection=Depends(get_data_collection_service),
    settings: Settings = Depends(get_settings),
    quota: GuestQuota = Depends(guest_quota("syllabus")),
) -> SyllabusResponse:
    session_id = extract_session_cookie(request)
    trace_context = {
        "endpoint": request.url.path,
        "feature": "syllabus_creation",
        "session_id": str(session_id) if session_id else None,
        "query_extracts_count": len(body.extracts),
        "documents_count": len(body.documents),
    }
    quota.consume()
    try:
        results = await tutor_manager(body, lang, settings, trace_context=trace_context)
    except Exception as e:
        quota.refund()
        logger.error(f"Syllabus generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Syllabus generation failed, please retry",
                "code": "SYLLABUS_GENERATION_FAILED",
            },
        )

    # the final syllabus is the last item: the only one in single-pass mode,
    # the PedagogicalEngineerAgent's in the legacy 3-agent chain
    final_syllabus = results[-1]

    message_id = await data_collection.register_syllabus_data(
        session_id=session_id,
        input_data=body,
        agent_answer=final_syllabus.content,
        feature="syllabus_creation",
    )

    return SyllabusResponse(
        syllabus=results,
        documents=body.documents,
        extracts=body.extracts,
        syllabus_message_id=message_id,
    )


@router.post("/syllabus/feedback", response_model=SyllabusResponse)
async def handle_syllabus_feedback(
    request: Request,
    body: SyllabusFeedback,
    data_collection=Depends(get_data_collection_service),
    settings: Settings = Depends(get_settings),
):
    session_id = extract_session_cookie(request)

    try:
        syllabus = await apply_feedback(
            body,
            settings,
            trace_context={
                "endpoint": request.url.path,
                "feature": "syllabus_feedback",
                "session_id": str(session_id) if session_id else None,
            },
        )

        await data_collection.register_syllabus_data(
            session_id=session_id,
            input_data=body,
            agent_answer=syllabus,
            feature="syllabus_feedback",
        )

        return SyllabusResponse(
            syllabus=[SyllabusResponseAgent(content=syllabus)],
            documents=body.documents,
            extracts=body.extracts,
        )

    except Exception as e:
        logger.error("Error in chat schema: %s", describe_error(e))
        raise e


@router.post("/syllabus/user_update", response_model=MessageResponse)
async def register_syllabus_user_update(
    request: Request,
    body: SyllabusUserUpdate,
    data_collection=Depends(get_data_collection_service),
):
    session_id = extract_session_cookie(request)

    await data_collection.register_syllabus_update(
        session_id=session_id,
        syllabus_content=body.syllabus,
    )

    return {"message": "user update registered"}


@router.post("/syllabus/description")
async def baml_test(body: CourseDescriptionRequest) -> types.CourseDescription:
    b_response = await b.GenerateCourseDescription(
        mode=body.mode,
        metadata=body.course_metadata,
        context_text=body.context_text,
        output_language=body.course_metadata.output_language,
    )
    return b_response


@router.post("/syllabus/learning_objectives")
async def baml_learning_objectives(
    body: LearningObjectivesRequest,
) -> types.LearningObjectives:
    b_response = await b.GenerateLearningObjectives(
        description=body.description,
        mode=body.mode,
        metadata=body.course_metadata,
        context_text=body.context_text,
        output_language=body.course_metadata.output_language,
    )

    return b_response


@router.post("/syllabus/sustainability_integration")
async def integrate_sustainability(
    body: IntegrateSustainabilityRequest,
) -> types.SustainabilityIntegration:
    print(body.course_metadata)
    # session_id = request.headers.get("X-Session-ID")

    b_response = await b.IntegrateSustainability(
        description=body.description,
        objectives=body.objectives,
        metadata=body.course_metadata,
        sdg_resources=body.sdg_resources,
        mode=body.mode,
        output_language=body.course_metadata.output_language,
    )

    return b_response


@router.post("/syllabus/learning_outcomes")
async def baml_learning_outcomes(
    body: LearningOutcomesRequest,
) -> types.LearningOutcomes:
    b_response = await b.GenerateLearningOutcomes(
        metadata=body.course_metadata,
        output_language=body.course_metadata.output_language,
        sustainability_map=body.sustainability_map,
        objectives=body.objectives,
    )

    return b_response


@router.post("/syllabus/competency_map")
async def generate_competency_map(
    body: CompetencyMappingRequest,
) -> types.CompetencyMappings:
    # session_id = request.headers.get("X-Session-ID")

    competencies = await b.MapCompetencies(
        outcomes=types.LearningOutcomes(
            outcomes=[
                types.LearningOutcome(
                    number=outcome.number,
                    text=outcome.text,
                    related_objectives=[],  # This field is not used in the BAML function, so we can leave it empty
                    assessment_method="",  # This field is not used in the BAML function, so we can leave it empty
                )
                for outcome in body.outcomes
            ]
        ),
        output_language=body.output_language,
        greencomp_framework=body.framework,
    )

    return competencies


@router.post(
    "/api/generate",
    response_model=SyllabusGenerationResponse,
    responses=GUEST_QUOTA_RESPONSES,
)
async def generate_syllabus(
    body: SyllabusGenerationRequest,
    quota: GuestQuota = Depends(guest_quota("syllabus")),
):
    quota.consume()
    try:
        user_input = UserInput(
            metadata=body.course_metadata,
            mode=body.mode,
            rag_resources=body.rag_resources,
            # The orchestrator reads the user's description from
            # provided_description; an empty string means "none".
            provided_description=body.course_metadata.user_description or None,
        )
        orchestrator = SyllabusOrchestrator()
        output = await orchestrator.run(user_input, context_text=body.context)

        return {
            "status": "success",
            "validation": {
                "passed": True,
                "major_issues": [],
                "minor_issues": [],
                "suggestions": [],
            },  # Placeholder for validation results
            **output,
        }
    except Exception as e:
        quota.refund()
        error = describe_error(e)
        logger.error("Error generating syllabus: %s", error)
        return {
            "status": "error",
            "message": error,
            "validation": {
                "passed": False,
                "major_issues": [error],
                "minor_issues": [],
                "suggestions": [],
            },
        }
