import unittest
from unittest import mock
from unittest.mock import MagicMock

from fastapi import HTTPException

from src.app.services.security import check_api_key, get_user


class SecurityTests(unittest.TestCase):
    @mock.patch("src.app.services.security.session_maker")
    def test_check_api_key_true_when_active(self, session_maker_mock):
        session = MagicMock()
        # Simulate found key with is_active True
        session.execute.return_value.first.return_value = MagicMock(is_active=True)
        session_maker_mock.return_value.__enter__.return_value = session

        assert check_api_key("secret-key") is True

    @mock.patch("src.app.services.security.session_maker")
    def test_check_api_key_false_when_not_found(self, session_maker_mock):
        session = MagicMock()
        session.execute.return_value.first.return_value = None
        session_maker_mock.return_value.__enter__.return_value = session

        assert check_api_key("does-not-exist") is False

    @mock.patch("src.app.services.security.session_maker")
    def test_check_api_key_false_when_inactive(self, session_maker_mock):
        session = MagicMock()
        session.execute.return_value.first.return_value = MagicMock(is_active=False)
        session_maker_mock.return_value.__enter__.return_value = session

        assert check_api_key("inactive") is False


class GetUserTests(unittest.TestCase):
    @mock.patch(
        "src.app.services.security.check_api_key", new=mock.MagicMock(return_value=True)
    )
    def test_get_user_ok(self):
        assert get_user("header-key") == "ok"

    @mock.patch(
        "src.app.services.security.check_api_key",
        new=mock.MagicMock(return_value=False),
    )
    def test_get_user_unauthorized(self):
        with self.assertRaises(HTTPException) as ctx:
            get_user("bad-key")
        self.assertEqual(ctx.exception.status_code, 401)
