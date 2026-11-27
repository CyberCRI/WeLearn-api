import unittest
import uuid
from datetime import date
from unittest import mock

from fastapi.testclient import TestClient

from src.app.core.config import settings
from src.app.shared.domain.exceptions import AccessExpiredError
from src.app.user.infra.keycloak import KeycloakUnavailableError, KeycloakUser
from src.app.user.services import account, limited_access, session
from src.app.user.services.limited_access import (
    check_limited_access,
    has_expired,
    is_limited,
)
from src.main import app

TODAY = date(2026, 11, 18)

client = TestClient(app)


def user(identity_provider=None, access_until=None) -> KeycloakUser:
    return KeycloakUser(
        keycloak_id=uuid.uuid4(),
        identity_provider=identity_provider,
        access_until=access_until,
    )


class LimitedAccessTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._patch(limited_access, "today", return_value=TODAY)
        self._patch(limited_access.settings, "LIMITED_ACCESS_ENABLED", new=True)
        self._patch(
            limited_access.settings,
            "LIMITED_ACCESS_IDENTITY_PROVIDERS",
            new=["google"],
        )
        self._patch(limited_access.settings, "LIMITED_ACCESS_DAYS", new=183)
        self.set_attribute = self._patch(
            limited_access,
            "set_user_attribute_if_missing",
            new_callable=mock.AsyncMock,
            side_effect=lambda keycloak_id, name, value: value,
        )
        limited_access._first_login_dates.clear()
        self.addCleanup(limited_access._first_login_dates.clear)

    def _patch(self, target, name, **kwargs):
        patcher = mock.patch.object(target, name, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m


class IsLimitedTests(LimitedAccessTestCase):
    def test_google_is_limited(self):
        self.assertTrue(is_limited(user("google")))

    def test_password_login_is_limited(self):
        self.assertTrue(is_limited(user(None)))

    def test_university_login_is_not_limited(self):
        self.assertFalse(is_limited(user("univ-paris")))


class HasExpiredTests(LimitedAccessTestCase):
    def test_last_day_is_included(self):
        self.assertFalse(has_expired("2026-11-18"))

    def test_past_date(self):
        self.assertTrue(has_expired("2026-11-17"))

    def test_future_date(self):
        self.assertFalse(has_expired("2027-05-20"))

    def test_datetime_uses_its_date(self):
        self.assertFalse(has_expired("2026-11-18T08:00:00Z"))

    def test_unreadable_date_counts_as_expired(self):
        self.assertTrue(has_expired("next year"))


class CheckLimitedAccessTests(LimitedAccessTestCase):
    async def test_flag_off_checks_nothing(self):
        with mock.patch.object(
            limited_access.settings, "LIMITED_ACCESS_ENABLED", False
        ):
            await check_limited_access(user("google", "2020-01-01"))
        self.set_attribute.assert_not_awaited()

    async def test_university_login_is_never_checked(self):
        await check_limited_access(user("univ-paris", "2020-01-01"))
        self.set_attribute.assert_not_awaited()

    async def test_first_limited_login_stores_the_end_date(self):
        google_user = user("google")

        await check_limited_access(google_user)

        self.set_attribute.assert_awaited_once_with(
            google_user.keycloak_id, "welearn_access_until", "2027-05-20"
        )

    async def test_first_login_date_is_cached_until_the_token_has_it(self):
        google_user = user("google")

        await check_limited_access(google_user)
        await check_limited_access(google_user)

        self.set_attribute.assert_awaited_once()

    async def test_cache_expires(self):
        google_user = user("google")
        monotonic = self._patch(limited_access.time, "monotonic", return_value=0.0)
        await check_limited_access(google_user)

        monotonic.return_value = limited_access.FIRST_LOGIN_CACHE_SECONDS + 1
        await check_limited_access(google_user)

        self.assertEqual(self.set_attribute.await_count, 2)

    async def test_date_already_stored_in_keycloak_is_used(self):
        # the token predates the attribute, which an admin set to a past date
        self.set_attribute.side_effect = None
        self.set_attribute.return_value = "2026-01-01"

        with self.assertRaises(AccessExpiredError):
            await check_limited_access(user("google"))

    async def test_date_in_the_token_is_not_rewritten(self):
        await check_limited_access(user(None, "2027-01-01"))
        self.set_attribute.assert_not_awaited()

    async def test_expired_access_raises(self):
        with self.assertRaises(AccessExpiredError) as ctx:
            await check_limited_access(user("google", "2026-11-17"))
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.msg_code, "ACCESS_EXPIRED")
        self.assertIn("2026-11-17", ctx.exception.message)

    async def test_keycloak_down_on_first_login_lets_the_user_in(self):
        self.set_attribute.side_effect = KeycloakUnavailableError("down")
        google_user = user("google")

        await check_limited_access(google_user)
        # not cached: the next request tries again
        await check_limited_access(google_user)

        self.assertEqual(self.set_attribute.await_count, 2)


class VerifyAccountTokenTests(LimitedAccessTestCase):
    async def test_expired_account_token_raises(self):
        request = mock.MagicMock(headers={"Authorization": "Bearer a.b.c"})
        self._patch(
            account,
            "decode_keycloak_token",
            return_value=user("google", "2026-01-01"),
        )

        with self.assertRaises(AccessExpiredError):
            await account.verify_account_token(request)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class UserAndSessionApiTests(LimitedAccessTestCase):
    def setUp(self):
        super().setUp()
        self.get_session = self._patch(
            session,
            "get_or_create_session_sync",
            return_value=uuid.uuid4(),
        )

    def _call(self, keycloak_user):
        self._patch(
            session,
            "get_or_create_keycloak_user",
            new_callable=mock.AsyncMock,
            return_value=(uuid.uuid4(), keycloak_user, False),
        )
        return client.post(
            f"{settings.API_V1_STR}/user/user_and_session",
            headers={"X-API-Key": "test", "Authorization": "Bearer a.b.c"},
        )

    def test_expired_account_gets_403_access_expired(self):
        response = self._call(user("google", "2026-01-01"))

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["code"], "ACCESS_EXPIRED")
        self.get_session.assert_not_called()

    def test_account_within_its_access_gets_a_session(self):
        response = self._call(user("google", "2027-01-01"))

        self.assertEqual(response.status_code, 200)

    def test_university_account_gets_a_session(self):
        response = self._call(user("univ-paris", "2020-01-01"))

        self.assertEqual(response.status_code, 200)


class GetAccountAccessTests(LimitedAccessTestCase):
    async def test_limited_account(self):
        result = await limited_access.get_account_access(user("google", "2027-05-20"))

        self.assertEqual(
            result.model_dump(),
            {"limited": True, "access_until": date(2027, 5, 20), "days_left": 184},
        )

    async def test_last_day_counts_as_one_day_left(self):
        result = await limited_access.get_account_access(user(None, "2026-11-18"))

        self.assertEqual(result.days_left, 1)

    async def test_first_login_gets_the_date_just_stored(self):
        result = await limited_access.get_account_access(user("google"))

        self.assertEqual(result.access_until, date(2027, 5, 20))

    async def test_date_not_stored_yet(self):
        self.set_attribute.side_effect = KeycloakUnavailableError("down")

        result = await limited_access.get_account_access(user("google"))

        self.assertEqual(
            result.model_dump(),
            {"limited": True, "access_until": None, "days_left": None},
        )

    async def test_university_account_is_not_limited(self):
        result = await limited_access.get_account_access(
            user("univ-paris", "2027-05-20")
        )

        self.assertEqual(
            result.model_dump(),
            {"limited": False, "access_until": None, "days_left": None},
        )

    async def test_flag_off(self):
        with mock.patch.object(
            limited_access.settings, "LIMITED_ACCESS_ENABLED", False
        ):
            result = await limited_access.get_account_access(user("google"))

        self.assertFalse(result.limited)


@mock.patch(
    "src.app.shared.infra.security.check_api_key_sync",
    new=mock.MagicMock(return_value=True),
)
class AccessApiTests(LimitedAccessTestCase):
    def _get(self, headers):
        return client.get(
            f"{settings.API_V1_STR}/user/access",
            headers={"X-API-Key": "test", **headers},
        )

    def _token_for(self, keycloak_user):
        self._patch(account, "decode_keycloak_token", return_value=keycloak_user)

    def test_limited_account(self):
        self._token_for(user("google", "2027-05-20"))

        response = self._get({"Authorization": "Bearer a.b.c"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"limited": True, "access_until": "2027-05-20", "days_left": 184},
        )

    def test_expired_account_gets_403(self):
        self._token_for(user("google", "2026-01-01"))

        response = self._get({"Authorization": "Bearer a.b.c"})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["code"], "ACCESS_EXPIRED")

    def test_guest_gets_401_account_required(self):
        response = self._get({})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"]["code"], "ACCOUNT_REQUIRED")
