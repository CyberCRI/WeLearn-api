import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.routing import APIRoute

# committed contract consumed by WeLearn-client, see `make openapi`
OPENAPI_SCHEMA_PATH = Path(__file__).resolve().parents[4] / "openapi.json"


def generate_operation_id(route: APIRoute) -> str:
    """
    Readable operation ids for generated clients: `<tag>-<function name>`
    (e.g. `search-get_documents_payload_by_ids`) instead of FastAPI's default
    `<name>_<path>_<method>`. Routes with an explicit `operation_id` (used by
    MCP) keep it.
    """
    if route.tags:
        return f"{route.tags[0]}-{route.name}"
    return route.name


def render_openapi_schema(app: FastAPI) -> str:
    """Serialize the schema deterministically so the committed file diffs cleanly."""
    return (
        json.dumps(app.openapi(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
