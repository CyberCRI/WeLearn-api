import unittest
from unittest import mock
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.app.shared.domain.exceptions import (
    EmailNotConfiguredError,
    EmailUnavailableError,
)
from src.app.user.api.router import (
    UNIVERSITY_REQUEST_ACCEPTED_MESSAGE,
    UNIVERSITY_REQUEST_RATE_LIMIT_PER_IP,
)
from src.app.user.domain.models import UniversityRequest
from src.app.user.services.university_request import (
    send_university_request,
    university_request_email,
)
from src.main import app

client = TestClient(app)
SERVICE = "src.app.user.services.university_request"


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
@mock.patch(f"{SERVICE}.send_email", new_callable=AsyncMock)
class UniversityRequestApiTests(unittest.TestCase):
    url = f"{settings.API_V1_STR}/user/university-request"

    def post(self, json):
        return client.post(self.url, json=json, headers={"X-API-Key": "test"})

    def test_request_is_emailed_to_the_team(self, send_mock):
        """Demande valide -> 202, email à l'équipe avec Reply-To"""
        response = self.post(
            {
                "university": "  Université\nde Nantes ",
                "city": "Nantes",
                "email": " Jane@Example.org",
                "locale": "fr-FR",
            }
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json(), {"message": UNIVERSITY_REQUEST_ACCEPTED_MESSAGE}
        )
        kwargs = send_mock.call_args.kwargs
        self.assertEqual(kwargs["to"], settings.UNIVERSITY_REQUEST_TO)
        self.assertEqual(kwargs["reply_to"], "jane@example.org")
        self.assertEqual(
            kwargs["subject"],
            "WeLearn – request to add a university: Université de Nantes",
        )
        self.assertIn("City: Nantes", kwargs["body"])
        self.assertIn("Email: jane@example.org", kwargs["body"])
        self.assertIn("Language: fr", kwargs["body"])

    def test_email_is_optional(self, send_mock):
        """Sans email -> 202, pas de Reply-To"""
        response = self.post({"university": "UPEC", "city": "Créteil", "email": ""})

        self.assertEqual(response.status_code, 202)
        self.assertIsNone(send_mock.call_args.kwargs["reply_to"])
        self.assertIn("Email: not given", send_mock.call_args.kwargs["body"])

    def test_invalid_fields(self, send_mock):
        """Champs vides, trop longs ou email invalide -> 422, rien n'est envoyé"""
        cases = [
            {"university": "   ", "city": "Paris"},
            {"university": "UPEC", "city": ""},
            {"university": "UPEC"},
            {"university": "x" * 201, "city": "Paris"},
            {"university": "UPEC", "city": "Paris", "email": "not-an-email"},
        ]
        for json in cases:
            response = self.post(json)
            self.assertEqual(response.status_code, 422, json)
        send_mock.assert_not_called()

    def test_rate_limited_per_ip(self, send_mock):
        """Trop de demandes depuis une même IP -> 429 RATE_LIMITED + Retry-After"""
        for _ in range(UNIVERSITY_REQUEST_RATE_LIMIT_PER_IP):
            self.assertEqual(
                self.post({"university": "UPEC", "city": "Créteil"}).status_code, 202
            )

        response = self.post({"university": "UPEC", "city": "Créteil"})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["code"], "RATE_LIMITED")
        self.assertGreater(int(response.headers["retry-after"]), 0)
        self.assertEqual(send_mock.call_count, UNIVERSITY_REQUEST_RATE_LIMIT_PER_IP)

    def test_smtp_not_configured(self, send_mock):
        """SMTP non configuré -> 500 EMAIL_DISABLED"""
        send_mock.side_effect = EmailNotConfiguredError()

        response = self.post({"university": "UPEC", "city": "Créteil"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "EMAIL_DISABLED")

    def test_smtp_unavailable(self, send_mock):
        """Serveur SMTP injoignable -> 503 EMAIL_UNAVAILABLE"""
        send_mock.side_effect = EmailUnavailableError()

        response = self.post({"university": "UPEC", "city": "Créteil"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "EMAIL_UNAVAILABLE")


class UniversityRequestServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_email_content(self):
        subject, body = university_request_email(
            UniversityRequest(university="UPEC", city="Créteil")
        )
        self.assertEqual(subject, "WeLearn – request to add a university: UPEC")
        self.assertIn("University: UPEC", body)
        self.assertIn("Language: unknown", body)

    @mock.patch(f"{SERVICE}.send_email", new_callable=AsyncMock)
    async def test_log_has_no_email_address(self, send_mock):
        with self.assertLogs(level="INFO") as logs:
            await send_university_request(
                UniversityRequest(
                    university="UPEC", city="Créteil", email="jane@example.org"
                )
            )
        self.assertNotIn("jane@example.org", "\n".join(logs.output))
