from datetime import datetime, timezone

from src.app.shared.infra.mailer import send_email
from src.app.shared.utils.dependencies import get_settings
from src.app.shared.utils.logger import logger as logger_utils
from src.app.user.domain.models import UniversityRequest
from src.app.user.services.signup import email_fingerprint

logger = logger_utils(__name__)


def university_request_email(request: UniversityRequest) -> tuple[str, str]:
    """Subject and plain-text body of the email sent to the WeLearn team."""
    subject = f"WeLearn – request to add a university: {request.university}"
    body = "\n".join(
        [
            "Someone asked to add their university to the WeLearn login page.",
            "",
            f"University: {request.university}",
            f"City: {request.city}",
            f"Email: {request.email or 'not given'}",
            f"Language: {request.locale or 'unknown'}",
            f"Date: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        ]
    )
    return subject, body


async def send_university_request(request: UniversityRequest) -> None:
    """
    Email the request to UNIVERSITY_REQUEST_TO, with Reply-To set to the person's
    email when they gave one.
    """
    subject, body = university_request_email(request)
    await send_email(
        to=get_settings().UNIVERSITY_REQUEST_TO,
        subject=subject,
        body=body,
        reply_to=request.email,
    )
    logger.info(
        "University request sent. email_sha256=%s",
        email_fingerprint(request.email) if request.email else None,
    )
