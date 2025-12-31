from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.app.services.search import get_qdrant
from src.main import app


@pytest.fixture(scope="class")
def client():
    app.dependency_overrides[get_qdrant] = lambda: AsyncMock()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
