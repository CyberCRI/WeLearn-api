from pydantic import BaseModel, Field


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
