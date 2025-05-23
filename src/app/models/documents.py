from pydantic import BaseModel


class Collection_schema(BaseModel):
    corpus: str
    name: str
    lang: str
    model: str


class DocumentPayloadModel(BaseModel):
    document_corpus: str
    document_desc: str
    document_details: dict[str, list[dict] | list[str] | str | int | float | None]
    document_id: str
    document_lang: str
    document_sdg: list[int]
    document_title: str
    document_url: str
    slice_content: str
    slice_sdg: int | None


class Document(BaseModel):
    score: float
    payload: DocumentPayloadModel
