"""Ponto de entrada FastAPI: ciclo de vida, CORS, handlers e registo de routers."""

import logging
from contextlib import asynccontextmanager

import spacy
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.config import get_settings
from backend.api.db.database import check_db_connection, create_tables
from backend.api.models.schemas import HealthResponse
from backend.api.routes import processes, upload, auth
from backend.api.utils.errors import (
    BaseAPIException,
    ConflictError,
    NotFoundError,
    ValidationError,
    UnauthorizedError,
)
from backend.api.utils.logging import setup_logging

# ========== Logging ==========
logger = logging.getLogger(__name__)

# ========== Configuração ==========
settings = get_settings()


# ========== Arranque e encerramento ==========
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    Ciclo de vida da aplicação FastAPI.

    Arranque:
    - Cria tabelas na base de dados
    - Carrega o modelo spaCy

    Encerramento:
    - Regista encerramento (ligações geridas pelo motor SQLAlchemy)
    """
    logger.info("=" * 80)
    logger.info("Starting application ARS - Automated Resume Screener...")
    logger.info("=" * 80)

    try:
        logger.info("Creating database tables...")
        create_tables()
        logger.info("Database tables created successfully.")

        logger.info("Checking database connection...")
        if not check_db_connection():
            logger.error("Database connection failed during startup -> Shutdown")
            raise RuntimeError("Database connection failed during startup")
        logger.info("Database connection successful.")

        logger.info("Loading spaCy model: %s...", settings.spacy_model)
        try:
            nlp = spacy.load(settings.spacy_model)
            fastapi_app.state.nlp_model = nlp
            logger.info("spaCy model loaded successfully: %s.", settings.spacy_model)
        except OSError as exc:
            logger.error(
                "spaCy model '%s' not found. Install with: python -m spacy download %s",
                settings.spacy_model,
                settings.spacy_model,
            )
            raise RuntimeError(f"Failed to load spaCy model: {exc!s}") from exc

        logger.info("Startup completed -> Application ready.")
        logger.info("=" * 80)

    except Exception as exc:
        logger.error("Fatal error during startup: %s", exc)
        raise

    yield

    logger.info("=" * 80)
    logger.info("Encerrando aplicação ARS")
    logger.info("=" * 80)


# ========== Aplicação FastAPI ==========
app = FastAPI(
    title="Automated Resume Screener",
    description="REST API for CV screening and ranking",
    version="2.0.0",
    lifespan=lifespan,
)

setup_logging(settings.log_level)

# CORS: origens permitidas a partir da configuração (lista)
if settings.allowed_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS configured for origins: %s", settings.allowed_origins_list)


# ========== Exception handlers ==========


async def validation_error_handler(request, exc: ValidationError):
    """
    Handler para ValidationError (400 Bad Request).

    Serviços lançam ValidationError para entrada inválida
    (ex.: título vazio, ficheiro não suportado, tamanho acima do limite).

    Resposta JSON: detail, error_code VALIDATION_ERROR.
    """
    logger.warning(
        "Validation error: %s | request: %s",
        exc.detail,
        request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": exc.detail,
            "error_code": "VALIDATION_ERROR",
        },
    )


async def not_found_error_handler(request, exc: NotFoundError):
    """
    Handler para NotFoundError (404 Not Found).

    Serviços lançam quando o recurso não existe
    (ex.: processo ou candidato inexistente).

    Resposta JSON: detail, error_code NOT_FOUND.
    """
    logger.warning(
        "Not found error: %s | request: %s",
        exc.detail,
        request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "detail": exc.detail,
            "error_code": "NOT_FOUND",
        },
    )


async def conflict_error_handler(request, exc: ConflictError):
    """
    Handler para ConflictError (409 Conflict).

    Serviços lançam quando a operação conflita com o estado atual
    (ex.: processar enquanto já está a processar, transição de estado inválida).

    Resposta JSON: detail, error_code CONFLICT.
    """
    logger.warning(
        "Conflict error: %s | request: %s",
        exc.detail,
        request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "detail": exc.detail,
            "error_code": "CONFLICT",
        },
    )


async def unauthorized_error_handler(request, exc: UnauthorizedError):
    """
    Handler para UnauthorizedError (401 Unauthorized).
    """
    logger.warning(
        "Unauthorized error: %s | request: %s",
        exc.detail,
        request.url.path,
    )


async def base_api_exception_handler(request, exc: BaseAPIException):
    """
    Handler para UnauthorizedError (401 Unauthorized).

    Chamador: Routes lancam quando token é inválido/expirado/revogado
    Exemplo: JWT não consegue descodificar, token em blacklist, user não existe

    Response:
    {
        "detail": "Invalid or expired token",
        "error_code": "UNAUTHORIZED"
    }
    """
    logger.warning(
        "Unauthorized error: %s | request: %s",
        exc.detail,
        request.url.path,
    )

    return JSONResponse(
        status_code=exc.HTTP_401_UNAUTHORIZED,
        content={
            "detail": exc.detail,
            "error_code": "UNAUTHORIZED",
        },
    )


app.add_exception_handler(ValidationError, validation_error_handler)
app.add_exception_handler(NotFoundError, not_found_error_handler)
app.add_exception_handler(ConflictError, conflict_error_handler)
app.add_exception_handler(UnauthorizedError, unauthorized_error_handler)
app.add_exception_handler(BaseAPIException, base_api_exception_handler)


# ========== Health check ==========
@app.get("/health", tags=["health"], response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check público (sem autenticação).

    Útil para balanceadores de carga, monitorização e pipelines de deploy.
    """
    try:
        db_status = "connected" if check_db_connection() else "disconnected"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Health deve responder mesmo com falha pontual na verificação à BD
        logger.warning("Health check -> DB error: %s", exc)
        db_status = "disconnected"

    nlp_status = (
        "loaded"
        if hasattr(app.state, "nlp_model") and app.state.nlp_model
        else "not loaded"
    )

    payload = HealthResponse(
        status="ok",
        version=app.version,
        database=db_status,
        nlp_model=nlp_status,
        environment=settings.app_env,
    )

    logger.debug("Health check response: %s", payload.model_dump())
    return payload


# ========== Raiz ==========
@app.get("/", tags=["root"])
async def root():
    """Metadados mínimos e ligações à documentação OpenAPI."""
    return {
        "message": "Welcome to Automated Resume Screener API",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


# ========== Routers ==========
app.include_router(auth.router)
app.include_router(processes.router)
app.include_router(upload.router)


logger.info("Routers registered: processes, upload")


if __name__ == "__main__":
    import uvicorn

    logger.info("Uvicorn starting on 0.0.0.0:8000")
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level="info",
    )
