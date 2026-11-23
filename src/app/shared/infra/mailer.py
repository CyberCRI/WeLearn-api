"""
Plain-text emails sent over SMTP (SMTP_* settings), with the standard library.

`EmailMessage` refuses CR/LF in header values, so text from a request can't add
headers; it still goes in the body rather than in a header when possible.
"""

import asyncio
import smtplib
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from src.app.shared.domain.exceptions import (
    EmailNotConfiguredError,
    EmailUnavailableError,
)
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils

logger = logger_utils(__name__)

SMTP_TIMEOUT_SECONDS = 10


def _send_sync(message: EmailMessage) -> None:
    settings = get_settings()
    try:
        with smtplib.SMTP(
            settings.SMTP_HOST, settings.SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS
        ) as smtp:
            if settings.SMTP_STARTTLS:
                smtp.starttls()
            if settings.SMTP_USER and settings.SMTP_PASSWORD:
                smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as e:
        logger.error("Email not sent: %s", e)
        raise EmailUnavailableError() from e


async def send_email(
    to: str, subject: str, body: str, reply_to: str | None = None
) -> None:
    """
    Send a plain-text email from SMTP_FROM to `to`. Raises EmailNotConfiguredError
    (500 EMAIL_DISABLED) without SMTP_HOST or SMTP_FROM, EmailUnavailableError (503
    EMAIL_UNAVAILABLE) when the server can't be reached or refuses it.
    """
    settings = get_settings()
    if not settings.SMTP_HOST or not settings.SMTP_FROM:
        logger.error("Email not sent: SMTP_HOST and SMTP_FROM must be set")
        raise EmailNotConfiguredError()

    message = EmailMessage()
    message["From"] = settings.SMTP_FROM
    message["To"] = to
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid()
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)

    await asyncio.to_thread(_send_sync, message)
