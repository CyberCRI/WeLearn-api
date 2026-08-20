import uuid
from enum import StrEnum, auto
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.http.models import Range, ScoredPoint
from qdrant_client.models import FieldCondition, Filter, MatchAny

from src.app.models.documents import Document
from src.app.utils.decorators import log_time_and_error_sync
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


class SearchOutput(BaseModel):
    search_message_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Identifier of the stored search event for analytics/tracking.",
    )
    docs: list[Document] | list[ScoredPoint] | None = Field(
        default=None,
        description="Search results returned by the endpoint.",
    )


class SDGFilter(BaseModel):
    sdg_filter: list[int] | None = Field(
        None,
        max_length=17,
        min_length=0,
        description="List of SDGs to filter the results",
        examples=[[1, 2, 3]],
    )


class SearchQuery(SDGFilter):
    query: str | list[str] | None = Field(
        ...,
        description="User query text.",
        examples=["How can schools reduce water waste?", "Compare carbon pricing and cap-and-trade: how do they work, and what are their pros and cons?"],
    )
    corpora: list[str] | None = Field(
        default=None,
        description="Optional list of corpus names to restrict search scope.",
        examples=[["conversation", "wikipedia"]],
    )
    lang: list[str] | None = Field(
        default=None,
        description="Optional language filters to apply to results.",
        examples=[["en", "fr"]],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "How can schools reduce water waste?",
                    "corpora": ["conversation"],
                    "lang": ["en"],
                    "sdg_filter": [6],
                },
                {
                    "query": "Compare carbon pricing and cap-and-trade: how do they work, and what are their pros and cons?",
                    "corpora": ["conversation", "wikipedia"],
                    "lang": ["en", "fr"],
                    "sdg_filter": [3, 6],
                },
            ]
        }
    )


class EnhancedSearchQuery(SDGFilter):
    query: str | list[str] = Field(
        ...,
        description="Search query payload.",
    )
    corpora: tuple[str, ...] | None = Field(
        default=None,
        description="Optional corpus tuple used to scope the search.",
    )
    lang: list[str] | None = Field(
        default=None,
        description="Optional language filters applied by the search backend.",
    )
    nb_results: int = Field(
        default=30,
        ge=1,
        description="Maximum number of results to return.",
    )
    subject: str | None = Field(
        default=None,
        description="Optional subject used to flavor/re-rank semantic search.",
    )
    influence_factor: float = Field(
        default=2,
        description="Subject influence factor for embedding flavoring.",
    )
    relevance_factor: float = Field(
        default=1,
        description="Relevance scaling factor used in ranking.",
    )
    concatenate: bool = Field(
        default=True,
        description="Whether to concatenate intermediate query strings before embedding.",
    )
    readability: Range | float | None = Field(
        default=None,
        description="Optional readability filter (range or score) applied during search.",
    )


class ContextType(StrEnum):
    INTRODUCTION = auto()
    TARGET = auto()
    SUBJECT = auto()


class FilterDefinition(BaseModel):
    key: str
    value: list[str] | Range | float | tuple[str, ...] | list[int] | None


class SearchFilters:
    def __init__(self, filters: list[FilterDefinition] | None) -> None:
        self.dict_filters: dict = {}
        self.filters = filters
        if not self.filters:
            self.filters = []
        for filter_item in self.filters:
            self.dict_filters[filter_item.key] = filter_item.value

    @log_time_and_error_sync
    def build_filters(self) -> Filter | None:
        if not self.filters:
            return None

        qdrant_filter = []
        for key, values in self.dict_filters.items():
            if not values:
                continue
            if isinstance(values, Range):
                qdrant_filter.append(
                    FieldCondition(
                        key=key,
                        range=values,
                    )
                )
            else:
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
