from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.app.shared.domain.responses import ErrorResponse
from src.app.shared.infra.db import session_maker
from src.app.shared.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

router = APIRouter()


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@router.get(
    "/",
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


def _ping_db() -> None:
    with session_maker() as session:
        session.execute(text("SELECT 1"))


@router.get(
    "/db",
    summary="Perform a Database Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Database unavailable",
        }
    },
)
async def get_db_health() -> HealthCheck:
    """
    ## Perform a Database Health Check
    Runs `SELECT 1` against Postgres. Returns 200 (OK) when the database answers,
    503 (Service Unavailable) with code `DATABASE_UNAVAILABLE` otherwise.
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    try:
        await run_in_threadpool(_ping_db)
    except (SQLAlchemyError, OSError) as e:
        logger.error("Database health check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Database unavailable",
                "code": "DATABASE_UNAVAILABLE",
            },
        )
    return HealthCheck(status="OK")
