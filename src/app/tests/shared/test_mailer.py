import smtplib
import unittest
from unittest import mock

from src.app.shared.domain.exceptions import (
    EmailNotConfiguredError,
    EmailUnavailableError,
)
from src.app.shared.infra import mailer

MAILER = "src.app.shared.infra.mailer"


def make_settings(**overrides):
    values = {
        "SMTP_HOST": "smtp.example.org",
        "SMTP_PORT": 587,
        "SMTP_USER": "user",
        "SMTP_PASSWORD": "secret",
        "SMTP_STARTTLS": True,
        "SMTP_FROM": "WeLearn <welearn@example.org>",
    }
    values.update(overrides)
    return mock.MagicMock(**values)


@mock.patch(f"{MAILER}.smtplib.SMTP")
class SendEmailTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_with_starttls_and_login(self, smtp_cls):
        smtp = smtp_cls.return_value.__enter__.return_value
        with mock.patch(f"{MAILER}.get_settings", return_value=make_settings()):
            await mailer.send_email(
                "team@example.org", "Subject", "Body", reply_to="jane@example.org"
            )

        smtp_cls.assert_called_once_with(
            "smtp.example.org", 587, timeout=mailer.SMTP_TIMEOUT_SECONDS
        )
        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with("user", "secret")
        message = smtp.send_message.call_args.args[0]
        self.assertEqual(message["From"], "WeLearn <welearn@example.org>")
        self.assertEqual(message["To"], "team@example.org")
        self.assertEqual(message["Subject"], "Subject")
        self.assertEqual(message["Reply-To"], "jane@example.org")
        self.assertEqual(message.get_content().strip(), "Body")

    async def test_no_tls_nor_login_without_them(self, smtp_cls):
        smtp = smtp_cls.return_value.__enter__.return_value
        settings = make_settings(SMTP_STARTTLS=False, SMTP_USER=None)
        with mock.patch(f"{MAILER}.get_settings", return_value=settings):
            await mailer.send_email("team@example.org", "Subject", "Body")

        smtp.starttls.assert_not_called()
        smtp.login.assert_not_called()
        self.assertIsNone(smtp.send_message.call_args.args[0]["Reply-To"])

    async def test_not_configured(self, smtp_cls):
        settings = make_settings(SMTP_HOST=None)
        with mock.patch(f"{MAILER}.get_settings", return_value=settings):
            with self.assertRaises(EmailNotConfiguredError):
                await mailer.send_email("team@example.org", "Subject", "Body")
        smtp_cls.assert_not_called()

    async def test_smtp_error(self, smtp_cls):
        smtp_cls.side_effect = smtplib.SMTPConnectError(421, "down")
        with mock.patch(f"{MAILER}.get_settings", return_value=make_settings()):
            with self.assertRaises(EmailUnavailableError):
                await mailer.send_email("team@example.org", "Subject", "Body")

    async def test_network_error(self, smtp_cls):
        smtp_cls.side_effect = TimeoutError("timed out")
        with mock.patch(f"{MAILER}.get_settings", return_value=make_settings()):
            with self.assertRaises(EmailUnavailableError):
                await mailer.send_email("team@example.org", "Subject", "Body")
