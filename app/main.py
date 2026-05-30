"""FastAPI application factory and lifespan."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_cors_origins
from app.core.logging_config import setup_logging

logger = logging.getLogger("docmind")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    logger.info("Initializing DocMind API")
    try:
        from app.services.bootstrap import initialize_models

        initialize_models()
    except Exception:
        logger.exception("Startup failed")
        raise
    logger.info("DocMind API ready")
    yield
    logger.info("Shutting down DocMind API")


def create_app(
    lifespan_handler: Callable[[FastAPI], AsyncIterator[None]] | None = None,
) -> FastAPI:
    application = FastAPI(
        title="DocMind API",
        description="RAG Chatbot with FastAPI — public demo on Hugging Face Spaces",
        lifespan=lifespan_handler or lifespan,
    )

    origins = get_cors_origins()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Please try again later."},
        )

    application.include_router(router)
    return application


app = create_app()
