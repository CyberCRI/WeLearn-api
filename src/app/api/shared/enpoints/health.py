from json import JSONDecodeError

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.app.api.dependencies import ConfigDepend

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


@router.get(
    "/db",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
async def get_db_health(settings: ConfigDepend) -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    try:
        async with httpx.AsyncClient() as client:
            rep = await client.get(
                f"{settings.DATABASE_HEALTHCHECK_URL}/health",
                timeout=settings.HTTPX_TIMEOUT,
            )
            rep_json: dict = rep.json()

        resp_status = rep_json.get("status")
        return HealthCheck(status=resp_status)
    except JSONDecodeError:
        raise HTTPException(status_code=403)
