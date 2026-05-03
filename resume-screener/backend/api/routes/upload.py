"""Rotas FastAPI para upload de currículos em lote por processo."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from backend.api.db.database import get_db
from backend.api.models.schemas import (
    UploadFailure,
    UploadResponse,
)
from backend.api.services.candidate_service import CandidateService
from backend.api.services.process_service import ProcessService
from backend.api.utils.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/processes", tags=["upload"])


def get_process_service(db: Session = Depends(get_db)) -> ProcessService:
    """
    Dependência FastAPI para injetar ProcessService.

    Args:
        db: Sessão SQLAlchemy injetada

    Returns:
        Instância de ProcessService
    """
    return ProcessService(db)


def get_candidate_service(
    db: Session = Depends(get_db),
    process_service: ProcessService = Depends(get_process_service),
) -> CandidateService:
    """
    Dependência FastAPI para injetar CandidateService.

    Args:
        db: Sessão SQLAlchemy injetada
        process_service: ProcessService injetado

    Returns:
        Instância de CandidateService
    """
    return CandidateService(db, process_service)


# Endpoint para upload de currículo
@router.post(
    "/{process_id}/upload",
    status_code=status.HTTP_200_OK,
    response_model=UploadResponse,
)
async def upload_candidates(
    process_id: UUID,
    files: list[UploadFile] = File(...),
    candidate_service: CandidateService = Depends(get_candidate_service),
) -> UploadResponse:
    """
    Upload múltiplos ficheiros de CV para um processo.

    Endpoint: POST /api/processes/{process_id}/upload
    Content-Type: multipart/form-data

    Request:
    - process_id: UUID do processo (path parameter)
    - files: múltiplos ficheiros com extensão .pdf, .docx, .txt

    Responses:
    - 200 OK: UploadResponse com uploaded count e failed array
    - 400 Bad Request: Nenhum ficheiro fornecido
    - 404 Not Found: Processo não existe
    - 409 Conflict: Processo em estado incompatível (ex: processando)

    Response Format:
    {
        "process_id": "uuid",
        "uploaded": 3,
        "failed": [
            {"filename": "empty.pdf", "reason": "Extracted text too short"},
            {"filename": "invalid.txt", "reason": "File size exceeds 10 MB"}
        ]
    }

    Comportamento:
    - Upload é tolerante a falhas (ficheiros inválidos são ignorados)
    - Um ficheiro com sucesso marca processo como files_uploaded
    - Múltiplos uploads para mesmo processo são permitidos (acumula candidatos)
    - Ficheiros duplicados (mesmo conteúdo) geram registos separados (sem dedup)

    Args:
        process_id: UUID do processo (FastAPI valida formato UUID)
        files: Lista de UploadFile do formulário multipart
        candidate_service: Injetado via Depends

    Returns:
        UploadResponse com contagem e lista de falhas

    Raises:
        ValidationError: Se nenhum ficheiro fornecido (400)
        NotFoundError: Se processo não existir (404)
        ConflictError: Se processo em estado incompatível (409)
    """
    logger.info(
        "POST /api/processes/%s/upload | files received: %d",
        process_id,
        len(files),
    )

    # Validar que pelo menos um ficheiro foi fornecido
    if not files or len(files) == 0:
        logger.warning("No files provided for the process: %s.", process_id)
        raise ValidationError("At least one file must be provided.")

    # Lista para rastreio
    uploaded_count = 0
    failed_uploads: list[UploadFailure] = []

    # Processar cada ficheiro individualmente
    for file in files:
        try:
            size_info = file.size if hasattr(file, "size") else "unknown"
            logger.debug(
                "Processing upload for file: %s | size:%s bytes",
                file.filename,
                size_info,
            )

            # CandidateService.save_file realiza:
            # 1. Valida (extension, MIME, size)
            # 2. Armazenamento em disco
            # 3. Criação de registo DB
            # 4. Transição de estado do processo

            candidate = candidate_service.save_file(process_id, file)

            uploaded_count += 1

            logger.info(
                "File uploaded successfully: %s | candidate: %s",
                file.filename,
                candidate.id,
            )

        except ValidationError as e:
            # Ficheiro inválido (ex: extension, MIME, size)
            logger.warning(
                "File upload failed: %s | Error: %s",
                file.filename,
                str(e),
            )

            failed_uploads.append(
                UploadFailure(
                    filename=file.filename,
                    reason=str(e),
                )
            )

        except ConflictError as e:
            # Conflito de estado do processo (ex: processando)
            logger.warning(
                "File upload conflict: %s | process: %s | error: %s",
                file.filename,
                process_id,
                str(e),
            )
            raise

        except NotFoundError as e:
            # Processo não encontrado
            logger.warning(
                "Process not found for file upload: %s | process: %s | error: %s",
                file.filename,
                process_id,
                str(e),
            )
            raise

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Tolerância por ficheiro; CandidateService pode lançar Exception genérica (disco/IO).
            logger.error(
                "Unexpected error during file upload: %s | process: %s | error: %s",
                file.filename,
                process_id,
                str(e),
            )
            failed_uploads.append(
                UploadFailure(
                    filename=file.filename,
                    reason="Unexpected error during upload.",
                )
            )

    # Validar que pelo menos um ficheiro foi carregado com sucesso
    if uploaded_count == 0:
        logger.warning(
            "All file uploads failed for process: %s | failures: %d",
            process_id,
            len(failed_uploads),
        )

        raise ValidationError(
            f"All files failed to upload. Details: {[f.reason for f in failed_uploads]}"
        )

    logger.info(
        "File upload completed for process: %s | success: %d | failed: %d",
        process_id,
        uploaded_count,
        len(failed_uploads),
    )

    # Construir response
    response = UploadResponse(
        process_id=process_id,
        uploaded=uploaded_count,
        failed=failed_uploads,
    )

    return response
