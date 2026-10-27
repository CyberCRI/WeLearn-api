from pydantic import BaseModel
from qdrant_client.http.models import ScoredPoint


class JourneySection(BaseModel):
    title: str
    content: str
    documents: list[ScoredPoint]


# keyed by journey section type (lowercase `JourneySectionType`)
FullJourney = dict[str, list[JourneySection]]
