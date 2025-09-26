from enum import StrEnum

from pydantic import BaseModel, Field
from qdrant_client.http.models import Range
from qdrant_client.models import FieldCondition, Filter, MatchAny

from src.app.utils.decorators import log_time_and_error_sync
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


class SDGFilter(BaseModel):
    sdg_filter: list[int] | None = Field(
        None,
        max_length=17,
        min_length=0,
        description="List of SDGs to filter the results",
        examples=[[1, 2, 3]],
    )


class SearchQuery(SDGFilter):
    query: str | list[str] | None
    corpora: list[str] | None = None


class EnhancedSearchQuery(SDGFilter):
    query: str | list[str]
    corpora: tuple[str, ...] | None = None
    nb_results: int = 30
    subject: str | None = None
    influence_factor: float = 2
    relevance_factor: float = 1
    concatenate: bool = True


class SearchFilters(BaseModel):
    slice_sdg: list[int] | None
    document_corpus: tuple[str, ...] | list[str] | None
    readability: Range | float | None

    @log_time_and_error_sync
    def build_filters(self) -> Filter | None:
        if not self.slice_sdg and not self.document_corpus:
            return None

        filters = {
            "slice_sdg": self.slice_sdg,
            "document_corpus": self.document_corpus,
            "document_details.readability": self.readability,
        }

        qdrant_filter = []
        for key, values in filters.items():
            if not values:
                continue

            qdrant_filter.append(
                FieldCondition(
                    key=key,
                    match=MatchAny(any=values),
                )
            )

        logger.debug("build_filters=%s", qdrant_filter)

        return Filter(must=qdrant_filter)


class SearchMethods(StrEnum):
    BY_SLICES = "by_slices"
    BY_DOCUMENT = "by_document"
