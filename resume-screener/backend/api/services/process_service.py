"""Serviço de gerenciamento de processos de screening."""

from typing import Optional
from uuid import UUID
from datetime import datetime, timezone
import logging

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from backend.api.db.models import Process, ProcessStatus
from backend.api.utils.errors import (
    NotFoundError,
    ConflictError,
    ValidationError,
)

logger = logging.getLogger(__name__)


class ProcessService:
    """
    Serviço de gerenciamento de processos de screening.

    Responsabilidades:
    - CRUD de processos
    - Validação de transições de estado
    - Persistência em PostgreSQL via SQLAlchemy

    Camada: Service Layer (abstrai BD, implementa lógica de negócio)
    Chamadores: routes/processes.py, routes/results.py
    Dependências: ORM Process model, custom exceptions
    """

    def __init__(self, db: Session):
        """
        Inicializa o serviço com a sessão do banco de dados.

        Args:
            db: Sessão do SQLAlchemy
        """
        self.db = db

    def create_process(self, title: str, jd_text: str) -> Process:
        """
        Criar novo processo de screening.

        Estado inicial: created
        Transição seguinte: files_uploaded (quando primeiro CV for uploaded)

        Args:
            title: Nome descritivo do processo (ex: "Backend Developer — Porto")
            jd_text: Texto completo da descrição da vaga

        Returns:
            Objeto Process com estado 'created', timestamps preenchidos

        Raises:
            ValidationError: Se title ou jd_text vazios
            SQLAlchemyError: Se falhar escrita no BD
        """

        if not title or not title.strip():
            logger.warning("Attempted to create process with empty title.")
            raise ValidationError("Title cannot be empty.")

        if not jd_text or not jd_text.strip():
            logger.warning("Attempted to create process with empty job description.")
            raise ValidationError("Job description cannot be empty.")

        try:
            now = datetime.now(timezone.utc)
            process = Process(
                title=title.strip(),
                jd_text=jd_text.strip(),
                status=ProcessStatus.CREATED,
                created_at=now,
                updated_at=now,
            )

            self.db.add(process)
            self.db.commit()
            self.db.refresh(process)

            logger.info(
                "Process created: %s | title: %s | JD length: %s chars",
                process.id,
                process.title,
                len(process.jd_text),
            )

            return process

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error("Error to create process in DB: %s", str(e))
            raise ValidationError(
                "Failed to create process due to database error."
            ) from e

    def list_processes(self, offset: int = 0, limit: int = 10) -> list[Process]:
        """
        Listar todos os processos ordenados por data (mais recentes primeiro).

        Ordem: created_at DESC (descending)
        Usado por: GET /api/processes

        Returns:
            Lista de objetos Process (pode estar vazia)

        Raises:
            SQLAlchemyError: Se falhar leitura no BD
        """

        try:
            processes = (
                self.db.query(Process)
                .order_by(Process.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            logger.debug("Listed %s processes from DB.", len(processes))
            return processes

        except SQLAlchemyError as e:
            logger.error("Error listing processes from DB: %s", str(e))
            raise ValidationError(
                "Failed to list processes due to database error."
            ) from e

    def get_process(self, process_id: UUID) -> Process:
        """
        Obter processo por ID.

        Args:
            process_id: UUID do processo

        Returns:
            Objeto Process

        Raises:
            NotFoundError: Se processo não existir (404)
            SQLAlchemyError: Se falhar leitura no BD
        """

        try:
            process = self.db.query(Process).filter(Process.id == process_id).first()

            if not process:
                logger.warning("Process not found: %s", process_id)
                raise NotFoundError(f"Process {process_id} not found.")

            logger.debug(
                "Retrieved process: %s | status: %s", process.id, process.status
            )
            return process

        except SQLAlchemyError as e:
            logger.error("Error retrieving process from DB: %s", str(e))
            raise ValidationError(
                "Failed to retrieve process due to database error."
            ) from e

    def update_status(
        self, process_id: UUID, new_status: str, error_message: Optional[str] = None
    ) -> Process:
        """
        Atualizar status de um processo com validação de máquina de estados.

        Máquina de estados:
            created → files_uploaded → processing → completed|failed
            qualquer → cancelled

        Transições inválidas geram ConflictError (409).

        Args:
            process_id: UUID do processo
            new_status: Novo status (deve ser string value de ProcessStatus enum)
            error_message: Mensagem de erro (obrigatória se new_status == "failed")

        Returns:
            Objeto Process com status atualizado

        Raises:
            NotFoundError: Se processo não existir (404)
            ConflictError: Se transição inválida (409)
            ValidationError: Se error_message vazio para status failed
            SQLAlchemyError: Se falhar escrita no BD
        """

        valid_statuses = {status.value for status in ProcessStatus}
        if new_status not in valid_statuses:
            logger.warning(
                "Invalid status value: %s. Valid statuses are: %s",
                new_status,
                valid_statuses,
            )
            raise ValidationError(f"Invalid status value: {new_status}.")

        process = self.get_process(process_id)

        if new_status == ProcessStatus.FAILED and not error_message:
            logger.warning(
                "Attempted to set status to 'failed' without error_message for process %s.",
                process_id,
            )
            raise ValidationError(
                "error_message is required when setting status to 'failed'."
            )

        current_status = process.status
        is_valid_transition = self._is_valid_transition(current_status, new_status)

        if not is_valid_transition:
            logger.warning(
                "Invalid transition: %s | %s → %s",
                process_id,
                current_status,
                new_status,
            )
            raise ConflictError(
                f"Invalid status transition: {current_status} → {new_status}."
            )

        try:
            process.status = new_status
            process.updated_at = datetime.now(timezone.utc)

            if new_status == ProcessStatus.COMPLETED:
                process.completed_at = datetime.now(timezone.utc)

            if new_status == ProcessStatus.FAILED:
                process.error_message = error_message
                process.completed_at = datetime.now(timezone.utc)

            self.db.commit()
            self.db.refresh(process)

            logger.info(
                "Status updated: %s | %s → %s", process.id, current_status, new_status
            )

            return process

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error("Error updating process status in DB: %s", str(e))
            raise ValidationError(
                "Failed to update process status due to database error."
            ) from e

    def _is_valid_transition(self, current_status: str, new_status: str) -> bool:
        """
        Validar se a transição de estado é permitida.

        Regras de transição:
        - created → files_uploaded
        - files_uploaded → processing
        - processing → completed ou failed
        - qualquer estado → cancelled
        - completed/failed → bloqueados (não podem sair destes estados)

        Args:
            current_status: Status atual (string)
            new_status: Status alvo (string)

        Returns:
            True se transição é válida, False caso contrário
        """

        if new_status == ProcessStatus.CANCELLED:
            if current_status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED):
                return False
            return True

        if current_status == ProcessStatus.CREATED:
            return new_status == ProcessStatus.FILES_UPLOADED

        if current_status == ProcessStatus.FILES_UPLOADED:
            return new_status == ProcessStatus.PROCESSING

        if current_status == ProcessStatus.PROCESSING:
            return new_status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED)

        return False

    def mark_files_uploaded(self, process_id: UUID) -> Process:
        """
        Marcar processo como tendo arquivos uploadados.

        Transição: created → files_uploaded
        Chamador: candidate_service.py quando primeiro arquivo é salvo com sucesso

        Args:
            process_id: UUID do processo

        Returns:
            Objeto Process com status files_uploaded

        Raises:
            NotFoundError: Se processo não existir
            ConflictError: Se transição inválida
        """
        return self.update_status(process_id, ProcessStatus.FILES_UPLOADED.value)

    def mark_processing(self, process_id: UUID) -> Process:
        """
        Marcar processo como em execução (screening pipeline iniciado).

        Transição: files_uploaded → processing
        Chamador: routes/results.py endpoint POST /processes/{id}/run

        Args:
            process_id: UUID do processo

        Returns:
            Objeto Process com status processing

        Raises:
            NotFoundError: Se processo não existir
            ConflictError: Se transição inválida
        """

        return self.update_status(process_id, ProcessStatus.PROCESSING.value)

    def mark_completed(self, process_id: UUID) -> Process:
        """
        Marcar processo como concluído com sucesso.

        Transição: processing → completed
        Chamador: screening_service.py quando pipeline termina sem erros

        Args:
            process_id: UUID do processo

        Returns:
            Objeto Process com status completed e completed_at preenchido

        Raises:
            NotFoundError: Se processo não existir
            ConflictError: Se transição inválida
        """

        return self.update_status(process_id, ProcessStatus.COMPLETED.value)

    def mark_failed(self, process_id: UUID, error_message: str) -> Process:
        """
        Marcar processo como falhado.

        Transição: processing → failed
        Chamador: screening_service.py quando pipeline encontra erro fatal

        Args:
            process_id: UUID do processo
            error_message: Descrição do erro (ex: "Spacy model failed to load")

        Returns:
            Objeto Process com status failed e error_message preenchido

        Raises:
            NotFoundError: Se processo não existir
            ConflictError: Se transição inválida
            ValidationError: Se error_message vazio
        """

        if not error_message or not error_message.strip():
            raise ValidationError("error_message cannot be empty.")

        return self.update_status(
            process_id, ProcessStatus.FAILED, error_message=error_message.strip()
        )

    def cancel_process(self, process_id: UUID) -> Process:
        """
        Cancelar processo.

        Transição: created|files_uploaded|processing → cancelled
        Chamador: futuro endpoint DELETE /processes/{id}

        Args:
            process_id: UUID do processo

        Returns:
            Objeto Process com status cancelled

        Raises:
            NotFoundError: Se processo não existir
            ConflictError: Se transição inválida (ex: já completed)
        """

        return self.update_status(process_id, ProcessStatus.CANCELLED.value)
