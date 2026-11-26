import asyncio
import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.app.search.domain.models import MAX_USER_QUERY_CHARS
from src.app.search.infra import sdg_classifier
from src.main import app

URL = f"{settings.API_V1_STR}/search/sdg_classify"
HEADERS = {"X-API-Key": "test"}


def prediction(code: str, value: float) -> dict:
    return {"prediction": value, "sdg": {"code": code, "name": f"Goal {code}"}}


@patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=MagicMock(return_value=True),
)
class SDGClassifyTests(TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.requests: list[httpx.Request] = []

    def classifier(self, handler):
        """The classifier answers with `handler`; the requests it got are kept."""

        async def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return await handler(request)

        patcher = patch.object(
            sdg_classifier,
            "_http_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(record)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def answer(self, predictions: list[dict]):
        async def handler(request):
            return httpx.Response(200, json={"predictions": predictions})

        self.classifier(handler)

    def post(self, body, headers=HEADERS):
        return self.client.post(URL, json=body, headers=headers)

    def assert_unavailable(self, response):
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["code"], "SDG_CLASSIFIER_UNAVAILABLE"
        )

    def test_best_sdgs_above_the_threshold(self):
        self.answer(
            [
                prediction("4", 0.05),
                prediction("13", 0.9),
                prediction("15", 0.1),
                prediction("6", 0.7),
                prediction("2", 0.12),
                prediction("12", 0.11),
            ]
        )

        response = self.post({"query": "forests and water"})

        self.assertEqual(response.status_code, 200)
        # 0.1 is not above the threshold, and only the 3 best are kept
        self.assertEqual(response.json(), {"sdgs": [13, 6, 2]})
        self.assertEqual(str(self.requests[0].url), settings.SDG_CLASSIFIER_URL)
        self.assertEqual(
            json.loads(self.requests[0].content), {"text": "forests and water"}
        )

    def test_forwards_no_client_header(self):
        self.answer([])

        self.client.post(
            URL,
            json={"query": "forests"},
            headers={**HEADERS, "User-Agent": "browser", "X-Forwarded-For": "1.2.3.4"},
        )

        sent = self.requests[0].headers
        self.assertNotIn("x-api-key", sent)
        self.assertNotIn("x-forwarded-for", sent)
        self.assertNotEqual(sent.get("user-agent"), "browser")

    def test_no_sdg_above_the_threshold(self):
        self.answer([prediction("4", 0.06), prediction("9", 0.1)])

        response = self.post({"query": "linear algebra"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"sdgs": []})

    def test_no_prediction(self):
        self.answer([])

        self.assertEqual(self.post({"query": "linear algebra"}).json(), {"sdgs": []})

    def test_timeout(self):
        async def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        self.classifier(handler)

        self.assert_unavailable(self.post({"query": "forests"}))

    def test_slow_answer(self):
        async def handler(request):
            await asyncio.sleep(1)
            return httpx.Response(200, json={"predictions": []})

        self.classifier(handler)

        with patch.object(
            sdg_classifier.settings, "SDG_CLASSIFIER_TIMEOUT_SECONDS", 0.05
        ):
            self.assert_unavailable(self.post({"query": "forests"}))

    def test_connection_error(self):
        async def handler(request):
            raise httpx.ConnectError("refused", request=request)

        self.classifier(handler)

        self.assert_unavailable(self.post({"query": "forests"}))

    def test_server_error(self):
        async def handler(request):
            return httpx.Response(500, text="Internal Server Error")

        self.classifier(handler)

        self.assert_unavailable(self.post({"query": "forests"}))

    def test_unusable_answers(self):
        for body in (
            "<html>maintenance</html>",
            json.dumps({"results": []}),
            json.dumps({"predictions": [prediction("18", 0.9)]}),
            json.dumps({"predictions": [prediction("SDG 4", 0.9)]}),
        ):
            with self.subTest(body=body):

                async def handler(request, body=body):
                    return httpx.Response(200, text=body)

                self.classifier(handler)

                self.assert_unavailable(self.post({"query": "forests"}))

    def test_query_too_long(self):
        self.answer([])

        response = self.post({"query": "a" * (MAX_USER_QUERY_CHARS + 1)})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"][-1], "query")
        self.assertEqual(self.requests, [])

    def test_longest_query_is_accepted(self):
        self.answer([])

        response = self.post({"query": "a" * MAX_USER_QUERY_CHARS})

        self.assertEqual(response.status_code, 200)

    def test_empty_or_missing_query(self):
        self.answer([])

        for body in ({"query": ""}, {}):
            with self.subTest(body=body):
                self.assertEqual(self.post(body).status_code, 422)
        self.assertEqual(self.requests, [])


class SDGClassifyApiKeyTests(TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.get_predictions = patch(
            "src.app.search.services.sdg_classifier.get_sdg_predictions",
            side_effect=AssertionError("must not run"),
        )
        self.get_predictions.start()
        self.addCleanup(self.get_predictions.stop)

    def test_missing_api_key(self):
        response = self.client.post(URL, json={"query": "forests"})

        # FastAPI's APIKeyHeader answers 403 when the header is missing
        self.assertEqual(response.status_code, 403)

    @patch(
        "src.app.shared.infra.security.check_api_key_sync",
        new=MagicMock(return_value=False),
    )
    def test_invalid_api_key(self):
        response = self.client.post(URL, json={"query": "forests"}, headers=HEADERS)

        self.assertEqual(response.status_code, 401)
