from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from src.app.core.lifespan import lifespan

LIFESPAN_MODULE = "src.app.core.lifespan"


class LifespanTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            QDRANT_HOST="http://qdrant",
            QDRANT_PORT=6333,
            EMBEDDING_MODELS_TO_PRELOAD=["model-a", "model-b"],
        )
        self.qdrant = MagicMock(close=AsyncMock())
        patches = {
            "get_settings": patch(
                f"{LIFESPAN_MODULE}.get_settings", return_value=self.settings
            ),
            "langsmith": patch(f"{LIFESPAN_MODULE}.configure_langsmith_tracing"),
            "init_chat": patch(f"{LIFESPAN_MODULE}.init_chat_model", new=AsyncMock()),
            "close_chat": patch(f"{LIFESPAN_MODULE}.close_chat_model", new=AsyncMock()),
            "qdrant": patch(
                f"{LIFESPAN_MODULE}.AsyncQdrantClient", return_value=self.qdrant
            ),
            "preload": patch(f"{LIFESPAN_MODULE}.preload_embedding_models"),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    async def test_preloads_the_configured_models_before_serving(self):
        app = SimpleNamespace(state=SimpleNamespace())

        async with lifespan(app):
            self.mocks["preload"].assert_called_once_with(["model-a", "model-b"])
            self.assertIs(app.state.qdrant, self.qdrant)

        self.qdrant.close.assert_awaited_once()
        self.mocks["close_chat"].assert_awaited_once()
