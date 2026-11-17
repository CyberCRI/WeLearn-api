"""
The per-IP limit on new guests, through the API and uvicorn's proxy headers
middleware as configured on k8s (FORWARDED_ALLOW_IPS = the Traefik pods' CIDR),
against an in-memory sqlite database.
"""

import unittest
import uuid
from datetime import datetime, timedelta
from unittest import mock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from welearn_database.data.models import InferredUser, Session

from src.app.core.config import settings
from src.app.shared.infra.rate_limit import SlidingWindowRateLimiter, client_ip
from src.app.tests.data_collection.sqlite_db import make_sqlite_session_maker
from src.app.user.services import account, guest_quota
from src.main import app

BASE = f"{settings.API_V1_STR}/user"
HEADERS = {"X-API-Key": "test"}
TRAEFIK_CIDR = "10.244.0.0/16"
TRAEFIK_POD = ("10.244.3.7", 50000)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class ClientIpBehindTraefikTests(unittest.TestCase):
    """What client_ip() returns once uvicorn trusts the Traefik pods."""

    def setUp(self):
        async def peer(request: Request) -> dict:
            return {"ip": client_ip(request)}

        echo = FastAPI()
        echo.get("/ip")(peer)
        self.trusting = ProxyHeadersMiddleware(echo, trusted_hosts=TRAEFIK_CIDR)

    def ip_seen(self, peer: tuple[str, int], forwarded_for: str | None) -> str:
        client = TestClient(self.trusting, client=peer)
        headers = {"X-Forwarded-For": forwarded_for} if forwarded_for else {}
        return client.get("/ip", headers=headers).json()["ip"]

    def test_user_ip_is_read_from_traefik(self):
        self.assertEqual(self.ip_seen(TRAEFIK_POD, "203.0.113.9"), "203.0.113.9")

    def test_forged_address_left_of_traefiks_is_ignored(self):
        self.assertEqual(
            self.ip_seen(TRAEFIK_POD, "1.2.3.4, 203.0.113.9"), "203.0.113.9"
        )

    def test_header_from_outside_the_cidr_is_not_trusted(self):
        self.assertEqual(self.ip_seen(("192.0.2.1", 50000), "1.2.3.4"), "192.0.2.1")

    def test_default_trust_ignores_the_header(self):
        # uvicorn's default FORWARDED_ALLOW_IPS is 127.0.0.1 only
        default = TestClient(
            ProxyHeadersMiddleware(self.trusting.app), client=TRAEFIK_POD
        )
        ip = default.get("/ip", headers={"X-Forwarded-For": "203.0.113.9"})
        self.assertEqual(ip.json()["ip"], TRAEFIK_POD[0])


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
@mock.patch(
    "src.app.shared.api.monitor_requests.register_endpoint", new=mock.MagicMock()
)
class NewGuestIpLimitApiTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.s_maker = make_sqlite_session_maker()
        self.clock = Clock()
        self.limiter = SlidingWindowRateLimiter(
            2, guest_quota.GUEST_SESSION_WINDOW_SECONDS, clock=self.clock
        )
        for patcher in (
            mock.patch("src.app.user.infra.queries.session_maker", self.s_maker),
            mock.patch.object(guest_quota, "_new_guest_ip_limiter", self.limiter),
            mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", True),
            mock.patch.object(guest_quota.settings, "GUEST_SESSIONS_PER_IP_HOURLY", 2),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.engine.dispose)
        self.client = TestClient(
            ProxyHeadersMiddleware(app, trusted_hosts=TRAEFIK_CIDR),
            client=TRAEFIK_POD,
        )

    def new_guest(self, ip: str = "203.0.113.9", **headers: str):
        return self.client.post(
            f"{BASE}/user_and_session",
            headers={**HEADERS, "X-Forwarded-For": ip, **headers},
        )

    def guests_in_db(self) -> int:
        with self.s_maker() as s:
            return s.query(InferredUser).count()

    def test_new_guests_past_the_limit_get_429(self):
        self.assertEqual(self.new_guest().status_code, 200)
        self.assertEqual(self.new_guest().status_code, 200)

        refused = self.new_guest()

        self.assertEqual(refused.status_code, 429)
        self.assertEqual(refused.json()["detail"]["code"], "RATE_LIMITED")
        self.assertEqual(refused.headers["Retry-After"], "3600")
        self.assertEqual(self.guests_in_db(), 2)

    def test_limit_comes_back_after_the_window(self):
        self.new_guest()
        self.new_guest()
        self.clock.now += guest_quota.GUEST_SESSION_WINDOW_SECONDS

        self.assertEqual(self.new_guest().status_code, 200)

    def test_another_ip_is_not_affected(self):
        self.new_guest()
        self.new_guest()

        self.assertEqual(self.new_guest(ip="198.51.100.4").status_code, 200)

    def test_known_session_is_not_counted(self):
        session_id = self.new_guest().json()["session_id"]
        self.new_guest()

        # page loads with the cookie keep working once the IP is used up
        for _ in range(3):
            renewed = self.new_guest(**{"X-Session-Id": session_id})
            self.assertEqual(renewed.status_code, 200)
            self.assertEqual(renewed.json()["session_id"], session_id)

    def test_expired_guest_session_is_renewed_without_counting(self):
        guest = uuid.uuid4()
        expired = uuid.uuid4()
        with self.s_maker() as s:
            s.add(InferredUser(id=guest))
            s.add(
                Session(
                    id=expired,
                    inferred_user_id=guest,
                    end_at=datetime.now() - timedelta(days=3),
                )
            )
            s.commit()
        self.new_guest()
        self.new_guest()

        renewed = self.new_guest(**{"X-Session-Id": str(expired)})

        self.assertEqual(renewed.status_code, 200)
        self.assertEqual(renewed.json()["user_id"], str(guest))

    @mock.patch("src.app.user.api.router.resolve_keycloak_user_and_session")
    def test_account_is_not_counted(self, resolve_account):
        resolve_account.return_value = (uuid.uuid4(), uuid.uuid4())
        self.new_guest()
        self.new_guest()

        response = self.new_guest(Authorization="Bearer token")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.limiter.peek("203.0.113.9")[0], 0)

    def test_no_limit_when_the_setting_is_0(self):
        with mock.patch.object(guest_quota.settings, "GUEST_SESSIONS_PER_IP_HOURLY", 0):
            for _ in range(3):
                self.assertEqual(self.new_guest().status_code, 200)
        self.assertEqual(len(self.limiter), 0)

    def test_no_limit_when_guest_restrictions_are_off(self):
        with mock.patch.object(account.settings, "GUEST_RESTRICTIONS_ENABLED", False):
            for _ in range(3):
                self.assertEqual(self.new_guest().status_code, 200)
        self.assertEqual(len(self.limiter), 0)
