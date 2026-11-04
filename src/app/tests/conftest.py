from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.app.search.services.search import get_qdrant
from src.app.shared.infra.rate_limit import reset_rate_limiters
from src.main import app


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    # limiters are module-level state: one test's hits must not limit the next
    reset_rate_limiters()
    yield
    reset_rate_limiters()


@pytest.fixture(scope="class")
def client():
    app.dependency_overrides[get_qdrant] = lambda: AsyncMock()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
