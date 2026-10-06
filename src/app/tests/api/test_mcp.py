import importlib.util
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from src import main
from src.app.core.config import settings
from src.app.models.documents import Document, DocumentPayloadModel
from src.app.search.services.search import get_search_service
from src.app.services.data_collection import get_data_collection_service


@asynccontextmanager
async def isolated_lifespan(app):
    yield


@pytest.fixture
def mcp_client(request, monkeypatch):
    monkeypatch.setattr(settings, "MCP_ENABLED", getattr(request, "param", True))
    monkeypatch.setattr("src.app.core.lifespan.lifespan", isolated_lifespan)
    spec = importlib.util.spec_from_file_location("mcp_test_main", main.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TestClient(module.app) as client:
        yield client


def initialize(client):
    client.headers["Accept"] = "application/json, text/event-stream"
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "integration-test", "version": "1.0"},
            },
        },
    )
    assert response.status_code == 200, response.text
    assert "result" in response.json()
    client.headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    assert response.status_code == 202, response.text


@pytest.mark.parametrize("mcp_client", [True, False], indirect=True)
def test_mcp_is_mounted_only_when_enabled(mcp_client):
    if settings.MCP_ENABLED:
        initialize(mcp_client)
    else:
        response = mcp_client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        )
        assert response.status_code == 404
        assert not any(route.path.startswith("/mcp") for route in mcp_client.app.routes)


def test_mcp_exports_only_allowed_operations(mcp_client):
    initialize(mcp_client)
    response = mcp_client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    )
    assert response.status_code == 200, response.text
    assert {tool["name"] for tool in response.json()["result"]["tools"]} == {
        "get_corpus_list",
        "search_by_document",
    }


@pytest.mark.parametrize("tool_name", ["get_corpus_list", "search_by_document"])
@pytest.mark.parametrize("api_key", [None, "invalid-key", "valid-key"])
def test_mcp_tools_forward_api_key(mcp_client, monkeypatch, tool_name, api_key):
    check_api_key = Mock(side_effect=lambda key: key == "valid-key")
    monkeypatch.setattr(
        "src.app.shared.infra.security.check_api_key_sync", check_api_key
    )
    collections = Mock(return_value=[("test-corpus", True, "Test Category")])
    monkeypatch.setattr(
        "src.app.search.api.router.get_collections_info_sync", collections
    )
    document = Document(
        score=0.9,
        payload=DocumentPayloadModel(
            document_id="12345678-1234-5678-1234-567812345678",
            document_corpus="test-corpus",
            document_desc="Clean water",
            document_details={},
            document_lang="en",
            document_sdg=[6],
            document_title="Clean water access",
            document_url="https://example.com/water",
            slice_content="Clean water access",
            slice_sdg=6,
        ),
    )
    search = AsyncMock(return_value=[document])
    data_collection = SimpleNamespace(register_search_data=AsyncMock(return_value=None))
    mcp_client.app.dependency_overrides[get_search_service] = lambda: SimpleNamespace(
        search_handler=search
    )
    mcp_client.app.dependency_overrides[get_data_collection_service] = (
        lambda: data_collection
    )
    initialize(mcp_client)
    if api_key is not None:
        mcp_client.headers["X-API-Key"] = api_key
    response = mcp_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": (
                    {"query": "clean water access", "nb_results": 1}
                    if tool_name == "search_by_document"
                    else {}
                ),
            },
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result.get("isError", False) is (api_key != "valid-key")
    if api_key is None:
        check_api_key.assert_not_called()
    else:
        check_api_key.assert_called_once_with(api_key)
    if api_key != "valid-key":
        collections.assert_not_called()
        search.assert_not_awaited()
        data_collection.register_search_data.assert_not_awaited()
    elif tool_name == "get_corpus_list":
        collections.assert_called_once_with()
        assert json.loads(result["content"][0]["text"]) == [
            {"name": "test-corpus", "category": "test_category", "is_active": True}
        ]
    else:
        search.assert_awaited_once()
        assert search.call_args.kwargs["qp"].query == "clean water access"
        data_collection.register_search_data.assert_awaited_once()
        payload = json.loads(result["content"][0]["text"])
        assert payload["docs"][0]["payload"]["document_title"] == "Clean water access"
