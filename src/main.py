import time

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client.http import exceptions as qdrant_exceptions
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.app.api.api_v1.api import api_router, api_tags_metadata
from src.app.api.shared.enpoints import health
from src.app.core.config import settings
from src.app.core.lifespan import lifespan
from src.app.middleware.monitor_requests import MonitorRequestsMiddleware
from src.app.services.security import get_user
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)

app = FastAPI(
    openapi_tags=api_tags_metadata,
    description=settings.DESCRIPTION,
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Middleware global monitoring
app.add_middleware(MonitorRequestsMiddleware)

# TODO: check this with JM
# @app.on_event("shutdown")
# async def shutdown_event():
#     await chatfactory.aclose()


@app.exception_handler(ResponseValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.exception_handler(qdrant_exceptions.UnexpectedResponse)
async def qdrant_unexpected(
    request: Request, exc: qdrant_exceptions.UnexpectedResponse
):
    """
    Custom exception handler for qdrant_client.http.exceptions.UnexpectedResponse
    """
    logger.error(
        "QDrant_unexpected error_details=%s status_code=%s",
        exc.reason_phrase,
        exc.status_code,
    )
    return JSONResponse(
        content=jsonable_encoder(
            {"message": exc.reason_phrase, "headers": exc.headers}
        ),
        status_code=exc.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(
        "HTTPException endpoint=%s status=%s detail=%s",
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return await http_exception_handler(request, exc)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    try:
        logger.debug("starting request=%s", request.url.path)
        start_time = time.time()

        response = await call_next(request)
        process_time = time.time() - start_time

        response.headers["X-Process-Time"] = str(process_time)
        logger.info(
            "add_process_time_header=%s endpoint=%s origin=%s status=%s",
            process_time,
            request.scope["path"],
            request.headers.get("origin"),
            response.status_code,
        )
        return response
    except RuntimeError as exc:
        if str(exc) == "No response returned." and await request.is_disconnected():
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        raise


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=settings.BACKEND_CORS_ORIGINS_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.api_route(path="/", tags=["root"], methods=["GET"])(settings.get_api_version)
app.include_router(health.router, prefix="/health", tags=["healthcheck"])
app.include_router(
    api_router,
    prefix=settings.API_V1_STR,
    dependencies=[Depends(get_user)],
)
