# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules for agents (mandatory)

- **Never install, add, upgrade or remove packages without explicit human approval.** This covers `poetry add/remove/update/install`, `pip install`, edits to the dependencies in `pyproject.toml` or `poetry.lock`, and system packages. If a task seems to need a new dependency, stop and ask, explaining why it is needed.
- **Never commit to or push to `main`.** Work on a feature branch. Commit and push only when a human asks, and never force-push. Changes reach `main` only through a pull request reviewed by a human.
- These rules are enforced in `.claude/settings.json`. Package commands and edits to `pyproject.toml` or `poetry.lock` ask for approval. The hook `.claude/hooks/protect-main.sh` blocks commits on `main`, pushes to `main`, and force-pushes.

## Commands

The project uses Poetry. The virtualenv lives in `.venv/`, and Python is 3.12–3.14; the Docker image uses 3.14.

```bash
poetry install                     # install deps (incl. dev group) — human approval required, see rules above
make generate-baml-poetry          # regenerate src/app/baml_client from src/app/baml_src (required before running or testing)
make run-poetry                    # generate BAML + uvicorn src.main:app --reload (port 8000)
make test-poetry                   # generate BAML + full pytest run with coverage gate (--cov-fail-under=82)
make lint                          # flake8, isort, black, lint-imports
make openapi-poetry                # regenerate openapi.json (the API/client contract) — commit it with the code change
```

- **Single test:** `poetry run pytest src/app/tests/search/test_search_api.py::SearchTests::test_search_items_no_query -q`. Add `-k pattern` to select by name.
- **Test env vars:** `pytest.ini` injects every required env var through `pytest-env`, so no `.env` is needed to run tests.
- **Lint fixes files:** `make lint` runs `isort src` and `black src` without `--check`, so it **rewrites files**. Run it before committing, then review the diff.
- **Dependency contracts:** `lint-imports` enforces the import rules in `pyproject.toml` (`[tool.importlinter]`) and fails the build on a violation.
- **Generated code:** `src/app/baml_client/` is generated and gitignored. Never edit it by hand. Change the `.baml` files under `src/app/baml_src/` and regenerate.

## Architecture

FastAPI service for the WeLearn platform: semantic search over a Qdrant vector store, RAG chat, and an LLM "tutor" that generates course syllabi. Postgres holds documents, users and sessions, analytics, and LangGraph checkpoints. Its ORM models come from the external `welearn-database` package (`welearn_database.data.models`); they are not defined in this repo.

### Request path

- **Entry point:** `src/main.py` builds the app and registers the middleware (`MonitorRequestsMiddleware`, process-time header, CORS) and the exception handlers.
- **API key:** `src/app/router.py` mounts every feature router under `settings.API_V1_STR` (`/api/v1`) with `Depends(get_user)`. Every API route therefore requires an `X-API-Key` header, which `shared/infra/security.py` checks against hashed keys in the DB. `/health` and `/` sit outside this router.
- **Startup (`src/app/core/lifespan.py`):**
  - creates the shared clients on `app.state`: `app.state.qdrant` (`AsyncQdrantClient`) and `app.state.llm` (`LLMProxy`, Mistral);
  - initialises the tutor's `ChatMistralAI` model;
  - configures LangSmith tracing.
- **API contract (`openapi.json`):**
  - The OpenAPI schema FastAPI generates is the single source of truth between this API and WeLearn-client. It is committed at the repo root, and the client generates its TypeScript types from it (`yarn gen:api` in WeLearn-client).
  - After changing a route, a Pydantic model it uses, or a `response_model`, run `make openapi-poetry` and commit `openapi.json`. `src/app/tests/shared/test_openapi_contract.py` fails when the committed file is stale.
  - The `openapi-contract` workflow runs `oasdiff breaking` on PRs that touch `openapi.json`. Breaking a field the client uses means changing the client in step: prefer additive changes.
  - Every route declares what it returns: a `response_model` (or a return annotation) with a dedicated Pydantic model, never a bare `dict` or an ORM object. Input and output models are separate. Responses that are not JSON (SSE, RIS text) declare their content type in `responses=`.
  - Operation ids are `<tag>-<function name>` (`shared/api/openapi.py::generate_operation_id`). An explicit `operation_id=` overrides it; keep the MCP ones.
  - JSON fields stay snake_case: the client and MCP consumers depend on it.
  - Errors are documented by `shared/domain/responses.py::ErrorResponse` (`{"detail": ...}`); raise domain errors with `detail={"message", "code"}` (see `shared/domain/exceptions.py`), the client reads `detail.code`.
- **MCP:** when `MCP_ENABLED`, `fastapi-mcp` exposes selected operations at `/mcp`. Operations are chosen by `operation_id`: `get_corpus_list` and `search_by_document`. Renaming those operation ids breaks the MCP interface.

### Feature-sliced layout (enforced)

`src/app/<feature>/` holds `api/` (router), `domain/` (Pydantic models), `services/` (logic) and `infra/` (DB queries and external clients).

The features are `search`, `chat`, `tutor`, `user`, `metric`, `micro_learning`, `bibliography` and `data_collection`.

`data_collection` has no router; other features call it to record analytics.

import-linter enforces three rules:

- `src.app.shared` and `src.app.core` must **not** import any feature.
- Only `src/app/router.py` may import a feature's `api` package.
- Feature routers are independent of each other.

Code that several features need goes in `shared/`. That includes domain exceptions, `documents.py` models, `infra/db.py` (`session_maker`), `llm_proxy.py`, and logger/decorator utils.

### Key cross-cutting pieces

- **Search:** `search/services/search.py::SearchService` embeds the query locally with HuggingFace models loaded from `../models/embedding/<model>/`, relative to the CWD. It then queries Qdrant and joins the hits with Postgres document data. Collections follow the pattern `collection_welearn_<lang>_<model>`.
- **Chat:**
  - **Stack:** `chat/services/abst_chat.py::AbstractChat` wraps LangChain `create_agent` with `ChatMistralAI`. The agent's search tool is in `chat/services/agent.py`.
  - **Agent memory:** conversations persist through LangGraph `AsyncPostgresSaver` in the `agent_related` schema, keyed by `thread_id`.
  - **Streaming:** endpoints return SSE through generators in `chat/services/chat_utils.py`.
- **Tutor:** there are two generation paths.
  - **BAML pipeline (`tutor/services/orchestrator.py::SyllabusOrchestrator`):**
    - runs the `b_agents/*` wrappers, one per BAML function, in phases;
    - is used by the `/tutor/syllabus/*` step endpoints and `/tutor/api/generate`;
    - uses BAML prompts in `src/app/baml_src/agents/tutor/*.baml`, with types in `types.baml` and the Mistral client in `clients.baml`;
    - code imports the **async** client: `from src.app.baml_client.async_client import b, types`.
  - **LangChain path (`tutor/services/tutor.py` + `syllabus.py`):**
    - used by `/tutor/syllabus`;
    - single pass when `TUTOR_SINGLE_PASS=True` (the default);
    - otherwise falls back to the legacy three-agent chain in `agents.py`.
- **Sessions and users:**
  - Anonymous sessions use the `x-session-id` cookie or header.
  - Optional Keycloak bearer tokens are verified in `user/infra/keycloak.py`.
  - `user/services/session.py` resolves a request to `(user_id, session_id)`.
  - `data_collection` only records when a campaign is active and the `Origin` matches `DATA_COLLECTION_ORIGIN_PREFIX`.
- **DB access:**
  - Queries are synchronous SQLAlchemy calls (`with session_maker() as s:`) in `*/infra/queries.py`.
  - Async routes must call them through `run_in_threadpool`; a few places don't yet, which is a known issue.
- **Settings:**
  - Values come from `src/app/core/config.py::Settings` (pydantic-settings, reads `.env`).
  - Use `get_settings()` from `shared/utils/dependencies.py` rather than instantiating `Settings`.
  - Several Azure fields are required at startup even though production uses Mistral only.

### Gotchas

- **Domain exceptions bypass `except Exception`.** The exceptions in `shared/domain/exceptions.py` currently subclass `BaseException`, not `Exception`, so `except Exception` does not catch them. Catch them explicitly.
- **Mock the API key in route tests.** Most tests are `unittest.TestCase` classes using `TestClient(app)`. API tests must patch `src.app.shared.infra.security.check_api_key_sync` (e.g. `MagicMock(return_value=True)`) and send `X-API-Key`.
- **Patch where a name is used.** Patch DB queries at the module that imports them, e.g. `src.app.user.api.router.get_user_bookmarks_sync`, not where they are defined.
- **Tutor files missing from coverage.** `.coveragerc` omits `tutor/services/tutor.py` and `tutor/services/agents.py`.

## Conventions

- **Logging:** use the project logger, `from src.app.shared.utils.logger import logger as utils_logger; logger = utils_logger(__name__)`, with `%s`-style arguments rather than f-strings.
- **Tests:** they live in `src/app/tests/<feature>/`, mirroring the source tree.
- **Commit messages:** follow conventional prefixes (`feat(scope):`, `chore:`, `refactor(...)`, `security:`).
