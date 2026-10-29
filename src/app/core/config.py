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
    Search
    **search results are grouped by url**
    - You can search by term
    - Filter based on metadata
    """

    BACKEND_CORS_ORIGINS_REGEX: str = CLIENT_ORIGINS_REGEX
    DATA_COLLECTION_ORIGIN_PREFIX: str
    SESSION_COOKIE_DOMAIN: str

    def get_api_version(self) -> dict:
        return {
            "title": self.PROJECT_NAME,
            "api_version": self.API_V1_STR,
            "semver": "0.1.0",
        }

    # AZURE ENV VARS
    AZURE_MISTRAL_API_BASE: str
    AZURE_MISTRAL_API_KEY: str

    AZURE_APIM_API_KEY: str
    AZURE_APIM_API_BASE: str

    # MISTRAL ENV VARS
    MISTRAL_API_KEY: str
    MISTRAL_LLM_MODEL_NAME: str

    # OPENAI_API FOR RAG METRICS (To be challenged)
    AZURE_API_KEY: str
    AZURE_API_BASE: str
    AZURE_API_VERSION: str

    LLM_MODEL_NAME: str
    LLM_TEMPERATURE: float
    # old tutor: one structured LLM call instead of the 3-agent chain (False = revert)
    TUTOR_SINGLE_PASS: bool = True
    ENV: str
    MCP_ENABLED: bool = True

    # LANGSMITH / LANGCHAIN TRACING
    # TBD: migrate Optional/Union annotations to PEP 604 shorthand (X | Y, X | None) in a dedicated refactor.
    LANGSMITH_TRACING_ENABLED: bool = False
    LANGSMITH_PROJECT: Optional[str] = None
    LANGSMITH_ENDPOINT: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None

    # PG
    PG_USER: Optional[str] = None
    PG_PASSWORD: Optional[str] = None
    PG_HOST: Optional[str] = None
    PG_PORT: Optional[str] = None
    PG_DATABASE: str
    PG_DRIVER: str

    QDRANT_HOST: str
    QDRANT_PORT: int

    # TIKA
    TIKA_URL_BASE: str

    # KEYCLOAK (front-end login; tokens are verified against the realm's JWKS)
    KEYCLOAK_URL: Optional[str] = None  # e.g. https://auth.example.org
    KEYCLOAK_REALM: Optional[str] = None
    # if set, the token's "azp" claim must match this front-end client id
    KEYCLOAK_CLIENT_ID: Optional[str] = None
    # sign-up: confidential client whose service account has the realm-management
    # roles "view-users" and "manage-users"
    KEYCLOAK_ADMIN_CLIENT_ID: Optional[str] = None
    KEYCLOAK_ADMIN_CLIENT_SECRET: Optional[str] = None
    # where the sign-up email link sends the user once done (needs KEYCLOAK_CLIENT_ID)
    KEYCLOAK_SIGNUP_REDIRECT_URI: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )


settings = Settings()  # type: ignore
