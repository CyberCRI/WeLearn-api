import os
from typing import Optional

from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo
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

    # the version of the running code, set by CI when it builds the image: the release tag
    # (v0.8.0) for a tag build, the commit sha otherwise. "dev" when run from the source
    APP_VERSION: str = "dev"

    def get_api_version(self) -> dict:
        return {
            "title": self.PROJECT_NAME,
            "api_version": self.API_V1_STR,
            "semver": self.APP_VERSION,
        }

    # MISTRAL ENV VARS
    MISTRAL_API_KEY: str
    MISTRAL_LLM_MODEL_NAME: str

    LLM_TEMPERATURE: float
    # old tutor: one structured LLM call instead of the 3-agent chain (False = revert)
    TUTOR_SINGLE_PASS: bool = True
    # syllabus v2 (/tutor/course/*): the longest one LLM call may take, retries of the
    # Mistral client included; past it the route answers 503 LLM_UNAVAILABLE
    TUTOR_LLM_TIMEOUT_SECONDS: float = 60
    # per session (or client IP without one), in the memory of each pod: LLM requests
    # running at once, and requests per window
    TUTOR_COURSE_IN_FLIGHT_PER_SESSION: int = 2
    TUTOR_COURSE_REQUESTS_PER_WINDOW: int = 30
    TUTOR_COURSE_WINDOW_SECONDS: int = 600
    # the same window per client IP too, since a caller picks its session id: room for the
    # teachers of one campus behind one address
    TUTOR_COURSE_REQUESTS_PER_IP_WINDOW: int = 120
    ENV: str

    # exposes get_corpus_list and search_by_document to MCP clients at /mcp
    MCP_ENABLED: bool = True

    # LANGSMITH / LANGCHAIN TRACING
    # TBD: migrate Optional/Union annotations to PEP 604 shorthand (X | Y, X | None) in a dedicated refactor.

    # traces of the LangChain calls sent to LangSmith, in LANGSMITH_PROJECT
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

    def postgres_conninfo(self) -> str:
        """libpq key=value connection string for psycopg 3 (the chat checkpointer).

        Not a postgresql:// URL: there the password would need URL-escaping, and a
        password with "%" or "#" breaks the connection (B-12). make_conninfo quotes
        the values and leaves out the ones that are not set.
        """
        return make_conninfo(
            "",
            host=self.PG_HOST,
            port=self.PG_PORT,
            dbname=self.PG_DATABASE,
            user=self.PG_USER,
            password=self.PG_PASSWORD,
        )

    QDRANT_HOST: str
    QDRANT_PORT: int

    # embedding models loaded at startup, from ../models/embedding/<name>/.
    # Other models still load on the first search that needs them. Override
    # with a JSON list, e.g. EMBEDDING_MODELS_TO_PRELOAD='[]' to turn it off
    EMBEDDING_MODELS_TO_PRELOAD: list[str] = ["granite-embedding-107m-multilingual"]

    # TIKA
    TIKA_URL_BASE: str

    # SDG classifier (Aurora, VU Amsterdam) for the search page's query. The API calls
    # it so the classifier gets the text but not the user's IP address or browser
    SDG_CLASSIFIER_URL: str = (
        "https://aurora-sdg.labs.vu.nl/classifier/classify/aurora-sdg-multi"
    )
    # the longest a call to the SDG classifier may take; past it /search/sdg_classify
    # answers 503 SDG_CLASSIFIER_UNAVAILABLE
    SDG_CLASSIFIER_TIMEOUT_SECONDS: float = 5

    # KEYCLOAK (front-end login; tokens are verified against the realm's JWKS)
    KEYCLOAK_URL: Optional[str] = None  # e.g. https://auth.example.org
    KEYCLOAK_REALM: Optional[str] = None
    # front-end client id: the token's "azp" claim must match it. Optional so the
    # app starts without Keycloak, but then bearer tokens are rejected with a 503
    KEYCLOAK_CLIENT_ID: Optional[str] = None
    # if set, the token's "aud" claim must contain it (needs an audience mapper
    # in Keycloak; access tokens' default "aud" is "account")
    KEYCLOAK_AUDIENCE: Optional[str] = None
    # sign-up: confidential client whose service account has the realm-management
    # roles "view-users" and "manage-users"
    KEYCLOAK_ADMIN_CLIENT_ID: Optional[str] = None
    KEYCLOAK_ADMIN_CLIENT_SECRET: Optional[str] = None
    # where the sign-up email link sends the user once done (needs KEYCLOAK_CLIENT_ID)
    KEYCLOAK_SIGNUP_REDIRECT_URI: Optional[str] = None
    # SMTP for the emails the API sends itself (requests to add a university).
    # Without SMTP_HOST and SMTP_FROM those endpoints answer 500 EMAIL_DISABLED
    SMTP_HOST: Optional[str] = None
    # SMTP server port (587 for STARTTLS submission)
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    # upgrade the SMTP connection with STARTTLS (off only for a local Mailpit)
    SMTP_STARTTLS: bool = True
    SMTP_FROM: Optional[str] = None  # e.g. "WeLearn <welearn@mg.lp-i.dev>"
    # where requests to add a university to the login page are sent
    UNIVERSITY_REQUEST_TO: str = "welearn@learningplanetinstitute.org"
    # account-only features (bookmarks, bibliography export, syllabus download)
    # answer 401 ACCOUNT_REQUIRED without a bearer token. Turn on with the
    # client's `auth` feature flag, per environment; off, a guest's cookie is enough
    GUEST_RESTRICTIONS_ENABLED: bool = False
    # daily guest quotas while GUEST_RESTRICTIONS_ENABLED is on (sliding 24h window
    # per anonymous user, counted in user_related.guest_feature_usage)
    GUEST_CHAT_DAILY_LIMIT: int = 5
    GUEST_SYLLABUS_DAILY_LIMIT: int = 1
    # changes asked on a generated syllabus, and LLM summaries of uploaded files
    GUEST_FEEDBACK_DAILY_LIMIT: int = 2
    GUEST_FILES_DAILY_LIMIT: int = 5
    # changes proposed for a sustainability connection (syllabus v2)
    GUEST_ADAPTATION_DAILY_LIMIT: int = 3
    # new guests per client IP and hour while GUEST_RESTRICTIONS_ENABLED is on
    # (0 = no limit). Only set it where FORWARDED_ALLOW_IPS lists the ingress
    # pods: otherwise the API sees the ingress' address and every user shares it
    GUEST_SESSIONS_PER_IP_HOURLY: int = 0
    # time-limited accounts: a login through one of LIMITED_ACCESS_IDENTITY_PROVIDERS
    # (Keycloak IdP aliases), or with an email and password, gets
    # LIMITED_ACCESS_DAYS from its first login, stored in the Keycloak user
    # attribute "welearn_access_until". Past it, account tokens answer 403
    # ACCESS_EXPIRED and the user goes back to guest mode. Other IdPs (Renater)
    # are not limited. Turn on only once the Keycloak client maps the
    # "identity_provider" session note and the "welearn_access_until" attribute
    # into the access token: without the first, every login counts as an
    # email/password one
    LIMITED_ACCESS_ENABLED: bool = False
    LIMITED_ACCESS_IDENTITY_PROVIDERS: list[str] = ["google"]
    LIMITED_ACCESS_DAYS: int = 183

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True
    )


settings = Settings()  # type: ignore
