from pydantic import BaseModel


class InstitutionData(BaseModel):
    institution: str
    role: str
