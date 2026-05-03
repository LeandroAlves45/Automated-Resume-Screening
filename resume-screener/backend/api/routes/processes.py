"""Rotas FastAPI para criação, listagem e consulta de processos de screening."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.api.db.database import get_db
from backend.api.models.schemas import (
    ProcessCreate,
    ProcessListResponse,
    ProcessResponse,
)
from backend.api.services.process_service import ProcessService
from backend.api.utils.errors import (
    NotFoundError,
    ValidationError,
)
from backend.api.routes.auth import get_current_user
from backend.api.db.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/processes", tags=["processes"])


def get_process_service(db: Session = Depends(get_db)) -> ProcessService:
    """
    Dependência FastAPI para injetar ProcessService.

    SessionLocal é criada por get_db dependency.
    ProcessService encapsula lógica de processos.

    Args:
        db: Sessão SQLAlchemy injetada

    Returns:
        Instância de ProcessService
    """
    return ProcessService(db)


# Endpoint para criar um novo processo
@router.post("", status_code=status.HTTP_201_CREATED, response_model=ProcessResponse)
async def create_process(
    request: ProcessCreate,
    process_service: ProcessService = Depends(get_process_service),
    current_user: User = Depends(get_current_user),
) -> ProcessResponse:
    """
    Criar novo processo de screening.
 
    Endpoint: POST /api/processes
    Autenticação: Requer JWT access token válido
 
    Request:
    {
        "title": "Backend Developer — Porto",
        "jd_text": "We are looking for a Python developer with 5+ years experience..."
    }
 
    Responses:
    - 201 Created: ProcessResponse (process_id, title, status, created_at)
    - 400 Bad Request: ValidationError (title ou jd_text vazios)
    - 401 Unauthorized: Token inválido ou expirado
    - 422 Unprocessable Entity: Pydantic validation failure
 
    Args:
        request: ProcessCreate schema com title e jd_text
        process_service: Injetado via Depends
        current_user: User autenticado via JWT (injetado via Depends)
 
    Returns:
        ProcessResponse com dados do processo criado
 
    Raises:
        ValidationError: Se title ou jd_text vazios (convertido a 400 por exception handler)
        UnauthorizedError: Se token inválido (capturado em get_current_user, convertido a 401)
    """
    logger.info(
        "POST /api/processes | user: %s | title: %s... | JD length: %d characters",
        current_user.email,
        request.title[:50],
        len(request.jd_text),
    )

    try:
        # ProcessService valida inputs (não-vazio, tamanho mínimo)
        process = process_service.create_process(
            title=request.title,
            jd_text=request.jd_text,
        )

        logger.info("Process created successfully: %s", process.id)

        # Converter ORM object -> Pydantic response
        response = ProcessResponse(
            process_id=process.id,
            title=process.title,
            status=process.status,
            created_at=process.created_at,
            updated_at=process.updated_at,
            completed_at=process.completed_at,
            error_message=process.error_message,
        )

        return response

    except ValidationError as e:
        logger.warning("Validation error while creating process: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error while creating process: %s", str(e))
        raise ValidationError(f"Failed to create process: {str(e)}") from e


@router.get("", response_model=ProcessListResponse)
async def list_processes(
    process_service: ProcessService = Depends(get_process_service),
    current_user: User = Depends(get_current_user),
    page: int = 0,
    limit: int = 10,
) -> ProcessListResponse:
    """
    Listar todos os processos ordenados por data (recentes primeiro).

    Endpoint: GET /api/processes
    Autenticação: Requer JWT access token válido

    Query Parameters:
    - page: int = 0 (índice da página para paginação)
    - limit: int = 10 (número de registos por página)

    Responses:
    - 200 OK: ProcessListResponse com array de ProcessResponse
    - 401 Unauthorized: Token inválido ou expirado
    - 400 Bad Request: Validação de query params falhou

    Args:
        process_service: Injetado via Depends
        current_user: User autenticado via JWT (injetado via Depends)
        page: Índice da página (default 0)
        limit: Número de processos por página (default 10)

    Returns:
        ProcessListResponse contendo lista de processos

    Raises:
        ValidationError: Se page < 0 ou limit fora do intervalo [1, 100]
        UnauthorizedError: Se token inválido (capturado em get_current_user, convertido a 401)
    """
    logger.info(
        "GET /api/processes | user: %s | page: %d | limit: %d",
        current_user.email,
        page,
        limit,
    )

    # Validar se o current user é um administrador

    # Validação básica de query params (página e limite não-negativos)
    if page < 0:
        raise ValidationError("Page number cannot be negative")

    if limit < 1 or limit > 100:
        raise ValidationError("Limit must be between 1 and 100")

    # SQL OFFSET = page * limit, LIMIT = limit
    offset = page * limit

    try:
        processes = process_service.list_processes(offset=offset, limit=limit)

        logger.info(
            "Listed %d processes successfully for user %s",
            len(processes),
            current_user.email,
        )

        # Converter ORM objects -> Pydantic responses
        process_response = [
            ProcessResponse(
                process_id=p.id,
                title=p.title,
                status=p.status,
                created_at=p.created_at,
                updated_at=p.updated_at,
                completed_at=p.completed_at,
                error_message=p.error_message,
            )
            for p in processes
        ]

        response = ProcessListResponse(processes=process_response)

        return response

    except Exception as e:
        logger.error("Unexpected error while listing processes: %s", str(e))
        raise ValidationError(f"Failed to list processes: {str(e)}") from e


@router.get("/{process_id}", response_model=ProcessResponse)
async def get_process(
    process_id: UUID,
    process_service: ProcessService = Depends(get_process_service),
    current_user: User = Depends(get_current_user),
) -> ProcessResponse:
    """
    Obter processo por ID.

    Endpoint: GET /api/processes/{process_id}
    Autenticação: Requer JWT access token válido

    Path Parameters:
    - process_id: UUID do processo

    Responses:
    - 200 OK: ProcessResponse com dados do processo
    - 401 Unauthorized: Token inválido ou expirado
    - 404 Not Found: Processo não existe

    Args:
        process_id: UUID do processo (FastAPI valida formato UUID)
        process_service: Injetado via Depends
        current_user: User autenticado via JWT (injetado via Depends)

    Returns:
        ProcessResponse com dados do processo

    Raises:
        NotFoundError: Se processo não existir (convertido a 404)
        UnauthorizedError: Se token inválido (capturado em get_current_user, convertido a 401)
    """
    logger.debug("GET /api/processes/%s | user: %s", process_id, current_user.email)

    try:
        process = process_service.get_process(process_id)

        logger.info(
            "Process %s retrieved successfully for user %s | status: %s",
            current_user.email,
            process_id,
            process.status,
        )

        response = ProcessResponse(
            process_id=process.id,
            title=process.title,
            status=process.status,
            created_at=process.created_at,
            updated_at=process.updated_at,
            completed_at=process.completed_at,
            error_message=process.error_message,
        )

        return response

    except NotFoundError as e:
        logger.warning("Process %s not found: %s", process_id, str(e))
        raise
    except Exception as e:
        logger.error(
            "Unexpected error while retrieving process %s: %s",
            process_id,
            str(e),
        )
        raise ValidationError(
            f"Failed to retrieve process {process_id}: {str(e)}"
        ) from e
