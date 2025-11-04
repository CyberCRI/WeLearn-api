from enum import StrEnum, auto

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
    readability: Range | float | None = None


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
