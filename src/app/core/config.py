import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
CLIENT_ORIGINS_REGEX: str = os.getenv("CLIENT_ORIGINS_REGEX", "").strip()


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "WeLearn"
    DESCRIPTION: str = """
    WeLearn API

    ## Search
    **search results are grouped by url**
    - You can search by term
    - Filter based on metadata
    """

    BACKEND_CORS_ORIGINS_REGEX: str = CLIENT_ORIGINS_REGEX

    def get_api_version(self, cls):
        return {
            "title": cls.PROJECT_NAME,
            "api_version": cls.API_V1_STR,
            "semver": "0.1.0",
        }

    QDRANT_HOST: str
    QDRANT_PORT: int
    CORPUS: str

    # MISTRAL
    MISTRAL_API_KEY: str

    # AZURE_API
    AZURE_LLAMA_31_8B_API_KEY: str
    AZURE_LLAMA_31_8B_API_BASE: str
    AZURE_LLAMA_31_70B_API_KEY: str
    AZURE_LLAMA_31_70B_API_BASE: str
    AZURE_LLAMA_31_405B_API_KEY: str
    AZURE_LLAMA_31_405B_API_BASE: str

    # OPENAI_API
    AZURE_API_KEY: str
    AZURE_API_BASE: str
    AZURE_API_VERSION: str
    # AZURE_API_TYPE: str

    # PG
    PG_USER: Optional[str] = None
    PG_PASSWORD: Optional[str] = None
    PG_HOST: Optional[str] = None
    PG_PORT: Optional[str] = None
    PG_DATABASE: str
    PG_DRIVER: str

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )


settings = Settings()  # type: ignore
