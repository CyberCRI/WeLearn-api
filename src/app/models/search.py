from pydantic import BaseModel, Field


class SearchFilter(BaseModel):
    sdg_filter: list[int] | None = Field(
        None,
        max_length=17,
        min_length=0,
        description="List of SDGs to filter the results",
        examples=[[1, 2, 3]],
    )


class SearchQuery(SearchFilter):
    query: str | list[str] | None
    corpora: list[str] | None = None


class EnhancedSearchQuery(SearchFilter):
    query: str | list[str]
    corpora: tuple[str, ...] | None = None
    nb_results: int = 30
    subject: str | None = None
    influence_factor: float = 2
    relevance_factor: float = 1
    concatenate: bool = True
