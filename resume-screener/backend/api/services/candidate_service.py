"""Serviço de candidatos: validação de uploads, armazenamento e operações na BD."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.api.config import get_settings
from backend.api.db.models import Candidate, ParseStatus, ProcessStatus
from backend.api.utils.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from backend.api.utils.validators import MAX_FILE_SIZE_BYTES, validate_upload_file

from .process_service import ProcessService

logger = logging.getLogger(__name__)


class CandidateService:
    """
    Serviço de gerenciamento de CVs (candidates).

    Responsabilidades:
    - Validação de ficheiros de upload (extension, MIME, tamanho)
    - Armazenamento de ficheiros no filesystem
    - Persistência de metadados em PostgreSQL
    - Transição de estado de processos (created → files_uploaded)

    Camada: Service Layer
    Chamadores: routes/upload.py
    Dependências: ProcessService, validators, ORM Candidate/Process, filesystem
    """

    def __init__(self, db: Session, process_service: ProcessService):
        self.db = db
        self.process_service = process_service
        self.settings = get_settings()

    def validate_file(self, file: UploadFile) -> dict[str, Any]:
        """
        Validar ficheiro de CV antes de salvar.

        Validações (nesta ordem):
        1. Extension contra whitelist [.pdf, .docx, .txt]
        2. MIME type via python-magic (detecta ficheiros renomeados)
        3. Tamanho máximo (10MB default, configurável)

        Args:
            file: FastAPI UploadFile do upload multipart/form-data

        Returns:
            dict com resultado (chaves compatíveis com save_file):
            is_valid, filename (sanitizado), file_type (extensão), size_bytes, error

        Raises:
            ValidationError: Falha de validação (subtipos do utils.errors)
        """
        try:
            raw = file.file.read()
            file.file.seek(0)
        except OSError as exc:
            logger.error("Failed to read uploaded file: %s", exc)
            raise ValidationError("Could not read uploaded file.") from exc

        try:
            result = validate_upload_file(
                raw,
                file.filename or "unnamed",
                MAX_FILE_SIZE_BYTES,
            )
        except ValidationError:
            if file.filename:
                logger.warning(
                    "Validation failed: %s | validator rejected upload",
                    file.filename,
                )
            raise

        return {
            "is_valid": True,
            "filename": result["sanitized_filename"],
            "file_type": result["extension"],
            "size_bytes": result["size_bytes"],
            "error": None,
        }

    # pylint: disable=too-many-branches,too-many-statements
    def save_file(
        self,
        process_id: UUID,
        file: UploadFile,
    ) -> Candidate:
        """
        Salvar ficheiro de CV no filesystem e criar registo no BD.

        Workflow:
        1. Validar ficheiro (extension, MIME, tamanho)
        2. Verificar que processo existe e está em estado válido
        3. Criar diretório de armazenamento se não existir
        4. Gerar stored_filename com UUID (previne path traversal)
        5. Copiar ficheiro para disco
        6. Criar registo Candidate no BD (parse_status: pending)
        7. Marcar processo como files_uploaded (transição de estado)

        Args:
            process_id: UUID do processo
            file: FastAPI UploadFile do formulário multipart

        Returns:
            Objeto Candidate com ficheiro salvo e BD persistido

        Raises:
            NotFoundError: Se processo não existir (404)
            ValidationError: Se ficheiro inválido (400)
            ConflictError: Se processo está em estado incompatível (409)
            RuntimeError: Falha de escrita em disco ou persistência inesperada

        Notes:
            - original_filename armazenado como uploaded (nunca usado em paths)
            - stored_filename é UUID-prefixado e sanitizado (segurança)
            - raw_text permanece None (preenchido por screening_service)
            - parse_status inicial é 'pending' (preenchido pela pipeline)
        """
        # 1. Validar ficheiro (levanta ValidationError se inválido)
        validation_result = self.validate_file(file)

        # 2. Verificar que processo existe
        try:
            process = self.process_service.get_process(process_id)
        except NotFoundError:
            logger.warning(
                "Try to upload for a inexistent process: %s",
                process_id,
            )
            raise

        # 3. Verificar estado do processo
        if process.status == ProcessStatus.PROCESSING:
            logger.warning(
                "Try to upload for a process in 'processing' state: %s",
                process_id,
            )
            raise ConflictError(
                "Cannot upload files while process is running (status: processing)"
            )

        # 4. Criar diretório de armazenamento se não existir
        storage_root = Path(self.settings.STORAGE_PATH)
        process_uploads_dir = storage_root / "processes" / str(process_id) / "uploads"

        try:
            process_uploads_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: %s", process_uploads_dir)

        except OSError as exc:
            logger.error(
                "Failed to create storage directory: %s | %s",
                process_uploads_dir,
                exc,
            )
            raise RuntimeError("Failed to create storage directory.") from exc

        # 5. Gerar stored_filename com UUID
        generated_uuid = uuid4()
        stored_filename = f"{generated_uuid}-{validation_result['filename']}"
        file_path = process_uploads_dir / stored_filename

        logger.debug(
            "Upload: %s -> %s | path: %s",
            file.filename,
            stored_filename,
            file_path,
        )

        # 6. Copiar ficheiro para disco
        try:
            with open(file_path, "wb") as disk_file:
                shutil.copyfileobj(file.file, disk_file)

            if not file_path.exists():
                raise OSError(
                    "Failed to write file to disk -> file does not exist after write."
                )

            logger.info(
                "File saved successfully: %s | size: %s bytes",
                stored_filename,
                validation_result["size_bytes"],
            )

        except OSError as exc:
            logger.error(
                "Failed to write file on disk: %s | path: %s",
                exc,
                file_path,
            )

            try:
                if file_path.exists():
                    file_path.unlink()

            except OSError as cleanup_error:
                logger.error(
                    "Failed to clean up partial file after write failure: %s",
                    cleanup_error,
                )
            raise RuntimeError(f"Failed to save file to disk: {exc}") from exc

        # 7. Criar registo Candidate na DB
        try:
            candidate = Candidate(
                process_id=process_id,
                name=validation_result["filename"],
                original_filename=file.filename,
                stored_filename=stored_filename,
                file_path=str(file_path),
                raw_text=None,
                parse_status=ParseStatus.PENDING,
                parse_error=None,
                created_at=datetime.now(timezone.utc),
            )

            self.db.add(candidate)
            self.db.flush()
            self.db.refresh(candidate)

            logger.info(
                "Candidate record created: %s | process: %s | file: %s",
                candidate.id,
                process_id,
                stored_filename,
            )

        except SQLAlchemyError as exc:
            self.db.rollback()
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.warning(
                        "Failed remove after DB error: %s",
                        stored_filename,
                    )
            except OSError as cleanup_error:
                logger.error(
                    "Failed to clean up file after DB error: %s",
                    cleanup_error,
                )

            logger.error("Error to save candidate to DB: %s", exc)
            raise RuntimeError(
                "Failed to save candidate record in database."
            ) from exc

        # 8. Transionar processo para files_uploaded
        try:
            self.process_service.mark_files_uploaded(process_id)
            self.db.commit()

            logger.info("Process marked as files_uploaded: %s", process_id)

        except ConflictError:
            self.db.rollback()
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.warning(
                        "File removed after process state conflict: %s",
                        stored_filename,
                    )

            except OSError as cleanup_error:
                logger.error(
                    "Failed to clean up file after process state conflict: %s",
                    cleanup_error,
                )

            logger.warning(
                "Process state conflict occurred for process: %s",
                process_id,
            )
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.db.rollback()
            try:
                if file_path.exists():
                    file_path.unlink()

            except OSError as cleanup_error:
                logger.error(
                    "Failed to clean up file: %s",
                    cleanup_error,
                )

            logger.error(
                "Unexpected error after creating candidate record: %s",
                exc,
            )
            raise RuntimeError(
                f"An unexpected error occurred after creating candidate: {exc}."
            ) from exc

        return candidate

    # pylint: enable=too-many-branches,too-many-statements

    def list_candidates(self, process_id: UUID) -> list[Candidate]:
        """
        Listar todos os candidatos de um processo.

        Ordem: created_at ASC (mais antigos primeiro)
        Usado por: GET /api/processes/{id}/results

        Args:
            process_id: UUID do processo

        Returns:
            Lista de Candidate objects (pode estar vazia)

        Raises:
            NotFoundError: Se processo não existir (404)
            ValidationError: Se falhar leitura no BD
        """
        try:
            self.process_service.get_process(process_id)
        except NotFoundError:
            logger.warning(
                "Try to list candidates for a inexistent process: %s",
                process_id,
            )
            raise

        try:
            candidates = (
                self.db.query(Candidate)
                .filter(Candidate.process_id == process_id)
                .order_by(Candidate.created_at.asc())
                .all()
            )

            logger.debug(
                "List %d candidates for process: %s",
                len(candidates),
                process_id,
            )

            return candidates

        except SQLAlchemyError as exc:
            logger.error(
                "Error to list candidates for process %s: %s",
                process_id,
                exc,
            )
            raise ValidationError(
                "Failed to list candidates from database."
            ) from exc

    def get_candidate(self, candidate_id: UUID) -> Candidate:
        """
        Obter candidato por ID.

        Args:
            candidate_id: UUID do candidato

        Returns:
            Objeto Candidate

        Raises:
            NotFoundError: Se candidato não existir (404)
            ValidationError: Se falhar leitura no BD
        """
        try:
            candidate = (
                self.db.query(Candidate)
                .filter(Candidate.id == candidate_id)
                .first()
            )

            if not candidate:
                logger.warning("Candidate not found: %s", candidate_id)
                raise NotFoundError("Candidate not found.")

            return candidate

        except SQLAlchemyError as exc:
            logger.error("Error to get candidate %s: %s", candidate_id, exc)
            raise ValidationError("Failed to retrieve candidate.") from exc

    def delete_candidate(self, candidate: Candidate) -> None:
        """
        Deletar ficheiro de CV do disco (para limpeza, GDPR, retenção).

        Não deleta o registo do BD (apenas o ficheiro físico).

        Args:
            candidate: Objeto Candidate com file_path preenchido

        Raises:
            ValidationError: Se ficheiro não existir ou erro ao deletar
        """
        if not candidate.file_path:
            logger.warning(
                "Candidate has no file path to delete: %s",
                candidate.id,
            )
            return

        file_path = Path(candidate.file_path)

        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(
                    "Deleted candidate file: %s | candidate: %s",
                    file_path,
                    candidate.id,
                )
            else:
                logger.warning(
                    "File already deleted or not found: %s | candidate: %s",
                    candidate.file_path,
                    candidate.id,
                )

        except OSError as exc:
            logger.error(
                "Failed to delete candidate file: %s | error: %s",
                candidate.file_path,
                exc,
            )
            raise ValidationError(
                f"Failed to delete candidate file: {exc}."
            ) from exc

    def get_file_path(self, candidate_id: UUID) -> str | None:
        """
        Obter caminho completo do ficheiro de um candidato.

        Usado internamente para leitura durante screening.
        Nunca exposto em respostas HTTP (segurança).

        Args:
            candidate_id: UUID do candidato

        Returns:
            Caminho completo do ficheiro ou None se não existir

        Raises:
            NotFoundError: Se candidato não existir
        """
        candidate = self.get_candidate(candidate_id)
        return candidate.file_path if candidate.file_path else None
