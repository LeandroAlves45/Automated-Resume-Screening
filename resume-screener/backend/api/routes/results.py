"""
Rotas de screening: execução e consulta de resultados.

Endpoints:
- POST /api/processes/{id}/run      → Dispara screening em background
- GET /api/processes/{id}/results   → Retorna ranking ou status

Autenticação: JWT obrigatória em ambos (get_current_user dependency).
Estado: máquina de estados validada antes de cada operação.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, status, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.api.db.database import get_db
from backend.api.db.models import User, ProcessStatus, Candidate
from backend.api.routes.auth import get_current_user
from backend.api.services.process_service import ProcessService
from backend.api.services.screening_service import ScreeningService
from backend.api.utils.errors import NotFoundError, ValidationError, ConflictError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/processes", tags=["screening"])


def get_nlp_model(request: Request) -> Any:
    """
    Extrai modelo spaCy carregado no startup.

    O modelo é armazenado em app.state durante o lifespan do FastAPI.
    Injetado aqui para estar disponível em todas as rotas que precisam NLP.

    Args:
        request: FastAPI Request context (acesso a app.state)

    Returns:
        spacy.Language model (en_core_web_sm)

    Raises:
        RuntimeError: Se modelo não foi carregado no startup (erro de configuração)
    """

    nlp = getattr(request.app.state, "nlp_model", None)
    if nlp is None:
        logger.error("spaCy model not found in app.state. Check startup events.")
        raise RuntimeError("NLP model not available. Check startup configuration.")
    return nlp


def get_screening_service(nlp: Any = Depends(get_nlp_model)) -> ScreeningService:
    """
    Factory para ScreeningService com injeção de dependências.

    Cria nova instância a cada requisição para isolamento de estado
    (estateless design).

    Args:
        nlp: spacy.Language injetado via get_nlp_model

    Returns:
        ScreeningService pronto para uso
    """
    return ScreeningService(nlp)


def validate_process_id(process_id: str) -> str:
    """
    Valida que process_id é um UUID válido.

    Args:
        process_id: String de UUID a validar

    Returns:
        process_id se válido

    Raises:
        ValidationError: Se formato não é UUID válido
    """

    try:
        # Tenta parsear como UUID (rejeita strings aleatórias)
        uuid.UUID(process_id)
        return process_id
    except ValueError as exc:
        raise ValidationError(
            f"Invalid process_id format: {process_id}. Must be a valid UUID."
        ) from exc


@router.post("/{process_id}/run", status_code=status.HTTP_200_OK)
async def run_screening(
    process_id: str = Depends(validate_process_id),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None,
    screening_service: ScreeningService = Depends(get_screening_service),
) -> dict:
    """
    Inicia screening de candidatos para um processo em background.

    Fluxo:
    1. Validar que processo existe e está em estado files_uploaded
    2. Verificar que há candidatos uploaded
    3. Disparar screening_service.run() em background
    4. Retornar imediatamente com status processing

    O cliente NÃO espera pela conclusão. Deve fazer polling em GET /results.

    Args:
        process_id: UUID do processo
        current_user: Utilizador autenticado (JWT válido)
        db_session: SQLAlchemy session para DB access
        background_tasks: FastAPI background task queue
        screening_service: Injetado, configurado com spaCy model

    Returns:
        {process_id, status: "processing"}

    Raises:
        404: Processo não existe
        409: Processo em estado incompatível (já processing, completed, etc)
        400: Nenhum candidato foi uploaded
        401: JWT inválido ou expirado (handled by get_current_user)
    """

    logger.info(
        "POST /run request for process %s by user %s",
        process_id,
        current_user.email,
    )

    process_service = ProcessService(db_session)

    try:
        # 1. Load e validar processo
        process = process_service.get_process(process_id)

        logger.debug(
            "Process %s loaded: state=%s, title=%s",
            process.id,
            process.status,
            process.title,
        )

        # 2. Validar estado: só avança se files_uploaded
        if process.status != ProcessStatus.FILES_UPLOADED:
            logger.warning(
                "Cannot run screening on process %s: current state is %s, requires files_uploaded",
                process.id,
                process.status,
            )
            raise ConflictError(
                detail=f"Cannot run screening: process is in '{process.status}' state."
                f"Expected: files_uploaded",
            )

        # 3. Verificar se há candidatos
        candidate_count = (
            db_session.query(Candidate)
            .filter(Candidate.process_id == process_id)
            .count()
        )

        if candidate_count == 0:
            logger.warning(
                "Cannot run screening on process %s: no candidates uploaded",
                process_id,
            )
            raise ValidationError(
                detail="No candidates uploaded for this process."
                "Please upload CVs before running screening.",
            )

        logger.info(
            "Running screening for process %s with %d candidate(s)",
            process_id,
            candidate_count,
        )

        # 4. Disparar screening em background
        background_tasks.add_task(screening_service.run, process_id, db_session)

        logger.info(
            "Screening task dispatched in background for process %s",
            process_id,
        )

        # 5. Responder imediatamente ao cliente
        return {
            "process_id": process_id,
            "status": "processing",
        }
    except (NotFoundError, ConflictError, ValidationError):
        # Exceptions customizadas propagam com status codes corretos
        raise
    except Exception as e:
        logger.error(
            "Unexpected error during POST /run for process %s: %s",
            process_id,
            str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while running screening.",
        ) from e


@router.get("/{process_id}/results")
async def get_results(
    process_id: str = Depends(validate_process_id),
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db),
    screening_service: ScreeningService = Depends(get_screening_service),
) -> dict:
    """
    Retorna resultados ou status do screening de um processo.

    Comportamento por estado do processo:

    - processing: retorna 202 Accepted com {status: "processing"}
    - completed: retorna 200 OK com ranking de candidatos + metadata
    - failed: retorna 200 OK com mensagem de erro
    - created|files_uploaded|cancelled: retorna 200 OK com status (sem resultados)

    Args:
        process_id: UUID do processo
        current_user: Utilizador autenticado (JWT válido)
        db_session: SQLAlchemy session para DB access
        screening_service: Injetado para acesso aos resultados

    Returns:
        dict com estrutura apropriada ao estado (vide detalhes abaixo)

    Status codes:
        202: Screening ainda em progresso
        200: Em qualquer outro estado (completed, failed, created, etc)
        404: Processo não existe
        401: JWT inválido ou expirado

    Response examples:

    202 Accepted (processing):
    {
        "status": "processing"
    }

    200 OK (completed):
    {
        "status": "completed",
        "summary": {
            "total": 10,
            "strong_matches": 2,
            "potential_matches": 4,
            "weak_matches": 4
        },
        "candidates": [
            {
                "rank": 1,
                "name": "Alice Silva",
                "total_score": 85.5,
                "category": "Strong Match",
                "breakdown": {
                    "skills_match": 100.0,
                    "experience_years": 80.0,
                    "education": 60.0,
                    "keyword_density": 85.2
                },
                "matched_skills": ["python", "fastapi", "postgresql"],
                "missing_skills": ["docker"],
                "experience_years_found": 6
            },
            ...
        ]
    }

    200 OK (failed):
    {
        "status": "failed",
        "error_message": "Screening failed: Invalid job description format"
    }

    200 OK (created, files_uploaded, cancelled):
    {
        "status": "created|files_uploaded|cancelled",
        "message": "Results not available in this process state"
    }
    """

    logger.info(
        "GET /results request for process %s by user %s",
        process_id,
        current_user.email,
    )

    process_service = ProcessService(db_session)

    try:
        # Load processo
        process = process_service.get_process(process_id)

        logger.debug(
            "Process %s status: %s, screening_service.get_results() called",
            process.id,
            process.status,
        )

        # Branching por estado
        if process.status == ProcessStatus.PROCESSING:
            logger.debug("Process %s is still processing. Returning 202.", process.id)
            # Retornar 202 Accepted com JSONResponse
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED, content={"status": "processing"}
            )

        # Para todos outros estados, ScreeningService.get_results() devolve dict apropriado
        results = screening_service.get_results(process_id)

        logger.info(
            "Results retrieved for process %s: status=%s",
            process_id,
            results.get("status"),
        )

        return results

    except NotFoundError:
        # Process não existe
        raise
    except Exception as e:
        logger.error(
            "Unexpected error during GET /results for process %s: %s",
            process_id,
            str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving results.",
        ) from e
