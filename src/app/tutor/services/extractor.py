from langchain_mistralai import ChatMistralAI  # type: ignore
from langsmith import traceable
from pydantic import BaseModel

from src.app.shared.infra.tracing import TraceName
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as utils_logger
from src.app.tutor.domain.models import ExtractorOutput, ExtractorOutputList, TutorLang
from src.app.tutor.services import tutor
from src.app.tutor.services.prompts import (
    extractor_system_prompt,
    extractor_user_prompt,
    summaries_schema,
)

logger = utils_logger(__name__)

# total characters of uploaded text sent to the LLM (~40k tokens)
MAX_DOCUMENTS_CHARS = 150_000


# Strictly typed LLM output: the public `ExtractorOutput.themes` is `list[Dict]`,
# which gives the model no usable schema for structured output.
class _Theme(BaseModel):
    theme: str
    reason: str


class _Extract(BaseModel):
    summary: str
    themes: list[_Theme]


class _Extracts(BaseModel):
    extracts: list[_Extract]


async def _get_model() -> ChatMistralAI:
    if tutor.chat_model is None:
        await tutor.init_chat_model(get_settings())
    return tutor.chat_model  # type: ignore[return-value]


def truncate_documents(contents: list[str], max_chars: int | None = None) -> list[str]:
    """
    Fit the documents' text into `max_chars` (default MAX_DOCUMENTS_CHARS).
    Documents shorter than an even share keep their whole text; the budget they
    leave goes to the longer ones, which are cut at the end.
    """
    if max_chars is None:
        max_chars = MAX_DOCUMENTS_CHARS
    budgets = [0] * len(contents)
    remaining = max_chars
    by_length = sorted(range(len(contents)), key=lambda i: len(contents[i]))
    for rank, index in enumerate(by_length):
        share = remaining // (len(contents) - rank)
        budgets[index] = min(len(contents[index]), share)
        remaining -= budgets[index]

    truncated = [content[:budget] for content, budget in zip(contents, budgets)]

    total = sum(len(content) for content in contents)
    if total > max_chars:
        logger.warning(
            "tutor_documents_truncated nb_documents=%s total_chars=%s max_chars=%s",
            len(contents),
            total,
            max_chars,
        )
    return truncated


def build_extractor_messages(documents: str, lang: TutorLang) -> list[tuple[str, str]]:
    return [
        (
            "system",
            extractor_system_prompt.format(json_schema=summaries_schema, lang=lang),
        ),
        ("user", extractor_user_prompt.format(documents=documents)),
    ]


@traceable(name=TraceName.FILES_SUMMARY_EXTRACTOR.value)
async def extract_files_summaries(
    documents: str, lang: TutorLang
) -> ExtractorOutputList:
    """Summarize each document and extract its main themes."""
    model = await _get_model()
    structured = model.with_structured_output(_Extracts)
    result = await structured.ainvoke(build_extractor_messages(documents, lang))
    if not isinstance(result, _Extracts):
        raise ValueError("Extractor returned an unexpected output")

    logger.debug("extract_files_summaries nb_extracts=%s", len(result.extracts))
    return ExtractorOutputList(
        extracts=[
            ExtractorOutput(
                summary=extract.summary,
                themes=[theme.model_dump() for theme in extract.themes],
            )
            for extract in result.extracts
        ]
    )
