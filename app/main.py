"""FastAPI application factory: lifespan (Redis, engine), request-id + timing middleware,
safe error handling (never leaks stack traces), router registration."""

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import admin, auth, chat, documents, health
from app.core.config import get_settings
from app.core.logging import get_logger, log_fields, request_id_ctx, setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.redis = None
    if settings.redis_url:
        from redis.asyncio import from_url

        app.state.redis = from_url(settings.redis_url, decode_responses=True)
    logger.info("app.startup", **log_fields(env=settings.env, model=settings.llm_model))
    yield
    if app.state.redis is not None:
        await app.state.redis.aclose()
    from app.db.session import engine

    await engine.dispose()
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Banking Regulatory RAG + Agent",
        description="Grounded Q&A over RBI / NPCI / NACH documents with citations and guardrails.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if settings.is_prod else "/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-ms"] = str(elapsed_ms)
        logger.info(
            "http.request",
            **log_fields(method=request.method, path=request.url.path, status=response.status_code, ms=elapsed_ms),
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("http.unhandled", **log_fields(path=request.url.path))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id_ctx.get()},
        )

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(auth.router, prefix=settings.api_prefix)
    app.include_router(chat.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(admin.router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/docs")

    return app


app = create_app()
