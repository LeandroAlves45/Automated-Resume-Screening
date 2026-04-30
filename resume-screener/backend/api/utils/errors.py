from typing import Optional

# =======================================================
# Exceção base -> Herança para todas as exceções de API
# =======================================================

class BaseAPIException(Exception):
  """
    Classe base para todas as exceções da API.
    
    Atributos:
        status_code: HTTP status code (400, 401, 404, 409, 500)
        error_code: Código interno (ex: 'PROCESS_NOT_FOUND')
        detail: Mensagem legível do erro
    
    FastAPI exception handlers devem catchear esta classe e retornar
    JSONResponse com status_code, detail e error_code.
    
    Exemplo de handler em main.py:
    
        @app.exception_handler(BaseAPIException)
        async def api_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "detail": exc.detail,
                    "error_code": exc.error_code
                }
            )
    """
  
  def __init__(
      self,
      detail: str,
      error_code: str,
      status_code: int = 500
  ):
    self.status_code = status_code
    self.error_code = error_code
    self.detail = detail

    # Chamar construtor da classe base
    super().__init__(self.detail)

# =======================================================
# Exceção 404 NOT FOUND
# =======================================================

class NotFoundError(BaseAPIException):
  """HTTP 404 — Recurso não encontrado.
    
    Usada quando:
    - GET /processes/{id} e processo não existe
    - GET /results e processo não existe
    - GET /candidates/{id} e candidato não existe
    
    Exemplo:
        if not process:
            raise NotFoundError(
                detail="Process with ID 123 not found",
                error_code="PROCESS_NOT_FOUND"
            )
    """
  
  def __init__(self, detail: str, error_code: str = "NOT_FOUND"):
    super().__init__(detail=detail, error_code=error_code, status_code=404)

# =======================================================
# Exceção 409 CONFLICT
# =======================================================

class ConflictError(BaseAPIException):
  """HTTP 409 — Conflito de estado.
    
    Usada quando operação viola máquina de estados ou lógica de negócio:
    - POST /run em processo já 'completed' ou 'failed'
    - POST /upload em processo em estado 'processing'
    - POST /upload em processo em estado 'completed'
    - Tentativa de criar recurso que já existe
    
    Exemplo:
        if process.status == ProcessStatus.PROCESSING:
            raise ConflictError(
                detail="Cannot upload files while screening is in progress",
                error_code="PROCESS_ALREADY_PROCESSING"
            )
    """
  
  def __init__(self, detail: str, error_code: str = "CONFLICT"):
    super().__init__(detail=detail, error_code=error_code, status_code=409)

# =======================================================
# Exceção 400 BAD REQUEST / VALIDATION
# =======================================================

class ValidationError(BaseAPIException):
  """HTTP 400 — Input inválido ou falha de validação.
    
    Usada quando:
    - Ficheiro excede MAX_FILE_SIZE_MB
    - MIME type não suportado
    - Extensão ficheiro não suportada
    - Processo criado com 0 candidatos, tentando rodar
    - Pydantic rejeita request body (normalmente automático)
    
    Exemplo:
        if len(file_bytes) > max_size:
            raise ValidationError(
                detail=f"File exceeds maximum size of {max_size} bytes",
                error_code="FILE_TOO_LARGE"
            )
    """
  
  def __init__(self, detail: str, error_code: str = "VALIDATION_ERROR"):
    super().__init__(detail=detail, error_code=error_code, status_code=400)

# =======================================================
# Exceção 401 UNAUTHORIZED
# =======================================================

class UnauthorizedError(BaseAPIException):
  """HTTP 401 — Autenticação falhada ou ausente.
    
    Usada quando:
    - JWT token ausente no header Authorization
    - JWT token inválido/expirado
    - Credenciais (email/password) incorretas no login
    - Token corrompido ou assinado com chave diferente
    
    Exemplo:
        if not token:
            raise UnauthorizedError(
                detail="Missing authorization token",
                error_code="MISSING_TOKEN"
            )
    """
  
  def __init__(self, detail: str, error_code: str = "UNAUTHORIZED"):
    super().__init__(detail=detail, error_code=error_code, status_code=401)

# =======================================================
# Exceção 403 FORBIDDEN
# =======================================================

class ForbiddenError(BaseAPIException):
  """HTTP 403 — Autenticado mas sem permissão.
    
    Usada quando:
    - Utilizador autenticado tenta acessar recurso de outro utilizador
    - Utilizador com role 'recruiter' tenta acessar área 'admin'
    - Ação requerida de um papel específico (ex: delete = admin only)
    
    Exemplo:
        if current_user.role != UserRole.ADMIN and not is_owner:
            raise ForbiddenError(
                detail="You do not have permission to delete this process",
                error_code="INSUFFICIENT_PERMISSIONS"
            )
    """
  
  def __init__(self, detail: str, error_code: str = "FORBIDDEN"):
    super().__init__(detail=detail, error_code=error_code, status_code=403)

# =======================================================
# Exceção 500 INTERNAL SERVER ERROR
# =======================================================

class InternalServerError(BaseAPIException):
  """HTTP 500 — Erro interno do servidor.
    
    Usada quando:
    - Exceção inesperada durante processamento
    - Falha de banco de dados não capturada
    - Falha de I/O (ficheiro não encontrado, permissões, etc)
    - spaCy model falha ao carregar
    - Pipeline v1.0 falha de forma catastrófica
    
    Nota: não expor detalhes técnicos ao cliente. Apenas logar internamente.
    
    Exemplo:
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise InternalServerError(
                detail="An unexpected error occurred. Please try again later.",
                error_code="INTERNAL_SERVER_ERROR"
            )
    """
  
  def __init__(
      self, 
      detail: str = "An unexpected error occurred.", 
      error_code: str = "INTERNAL_SERVER_ERROR"
  ):
    super().__init__(detail=detail, error_code=error_code, status_code=500)

# =======================================================
# Exceções específicas de domínio -> especialização de exceções base
# =======================================================

class ProcessNotFoundError(NotFoundError):
  """Processo específico não encontrado."""

  def __init__(self, process_id: str):
    detail = f"Process with ID {process_id} not found"
    super().__init__(detail=detail, error_code="PROCESS_NOT_FOUND")

class CandidateNotFoundError(NotFoundError):
  """Candidato específico não encontrado."""

  def __init__(self, candidate_id: str):
    detail = f"Candidate with ID {candidate_id} not found"
    super().__init__(detail=detail, error_code="CANDIDATE_NOT_FOUND")

class UserNotFoundError(NotFoundError):
  """Utilizador específico não encontrado."""

  def __init__(self, user_id: str):
    detail = f"User with ID {user_id} not found"
    super().__init__(detail=detail, error_code="USER_NOT_FOUND")

class ProcessAlreadyProcessingError(ConflictError):
  """Tentativa de operação conflitante quando o processo está em execução."""

  def __init__(self, process_id: str):
    detail = f"Process with ID {process_id} is already processing"
    super().__init__(detail=detail, error_code="PROCESS_ALREADY_PROCESSING")

class ProcessAlreadyCompletedError(ConflictError):
  """Tentativa de operação conflitante quando o processo já foi concluído."""

  def __init__(self, process_id: str):
    detail = f"Process with ID {process_id} has already been completed"
    super().__init__(detail=detail, error_code="PROCESS_ALREADY_COMPLETED")

class UploadNotAllowedInStateError(ConflictError):
  """HTTP 409 — Upload não permitido no estado atual do processo.

  Usada quando:
  - Tentativa de upload em processo 'processing'
  - Tentativa de upload em processo 'completed'
  - Tentativa de upload em processo 'failed'

  Exemplo:
      if process.status not in [ProcessStatus.CREATED, ProcessStatus.FILES_UPLOADED]:
          raise UploadNotAllowedInStateError(
              process_id=process.id,
              current_status=process.status.value
          )
  """

  def __init__(self, process_id: str, current_status: str):
    detail = f"Cannot upload files to process {process_id} in state '{current_status}'. Upload only allowed in 'created' or 'files_uploaded' states."
    super().__init__(detail=detail, error_code="UPLOAD_NOT_ALLOWED_IN_STATE")

class ScreeningNotAllowedInStateError(ConflictError):
  """HTTP 409 — Screening não pode ser executado no estado atual do processo.

  Usada quando:
  - Tentativa de rodar /run em processo 'created' (sem ficheiros)
  - Tentativa de rodar /run em processo já 'processing'
  - Tentativa de rodar /run em processo já 'completed'

  Exemplo:
      if process.status != ProcessStatus.FILES_UPLOADED:
          raise ScreeningNotAllowedInStateError(
              process_id=process.id,
              current_status=process.status.value
          )
  """

  def __init__(self, process_id: str, current_status: str):
    detail = f"Cannot run screening on process {process_id} in state '{current_status}'. Screening only allowed in 'files_uploaded' state."
    super().__init__(detail=detail, error_code="SCREENING_NOT_ALLOWED_IN_STATE")

class NoCandidatesError(ValidationError):
  """Tentativa de rodar screening sem candidatos associados ao processo."""

  def __init__(self, process_id: str):
    detail = f"Cannot run screening on process {process_id}: no candidates uploaded"
    super().__init__(detail=detail, error_code="NO_CANDIDATES_TO_SCREEN")

class FileTooLargeError(ValidationError):
  """Ficheiro excede o tamanho máximo permitido."""

  def __init__(self, filename: str, max_size_mb: int):
    detail = f"File '{filename}' exceeds maximum size of {max_size_mb} MB"
    super().__init__(detail=detail, error_code="FILE_TOO_LARGE")

class UnsupportedFileTypeError(ValidationError):
  """Extensão de ficheiro não suportada."""

  def __init__(self, filename: str, supported_types: list[str]):
    detail = f"File '{filename}' has unsupported type or extension. Supported types: {', '.join(supported_types)}"
    super().__init__(detail=detail, error_code="UNSUPPORTED_FILE_TYPE")

class InvalidMimeTypeError(ValidationError):
  """MIME type do ficheiro não suportado."""

  def __init__(self, filename: str, mime_types: str):
    detail = f"File '{filename}' has unsupported MIME type. Supported MIME types: {mime_types}"
    super().__init__(detail=detail, error_code="INVALID_MIME_TYPE")

class InvalidCredentialsError(UnauthorizedError):
  """Credenciais de login inválidas."""

  def __init__(self):
    detail = "Invalid email or password"
    super().__init__(detail=detail, error_code="INVALID_CREDENTIALS")

class EmailAlreadyRegisteredError(ConflictError):
  """Tentativa de registrar com email já existente."""

  def __init__(self, email: str):
    detail = f"Email '{email}' is already registered"
    super().__init__(detail=detail, error_code="EMAIL_ALREADY_REGISTERED")

class InvalidTokenError(UnauthorizedError):
  """Token JWT inválido ou expirado."""

  def __init__(self, reason: Optional[str] = None):
    detail = "Invalid or expired token"

    if reason:
      detail += f": {reason}"

    super().__init__(detail=detail, error_code="INVALID_TOKEN")

class MissingTokenError(UnauthorizedError):
  """Token JWT ausente no header Authorization."""

  def __init__(self):
    detail = "Missing authorization token"
    super().__init__(detail=detail, error_code="MISSING_TOKEN")

# =======================================================
# Exceção de pipeline -> Erros do pipeline
# =======================================================

class PipelineError(InternalServerError):
  """Pipeline falhou durante execução.
    
    Captura exceções de:
    - ResumeParser falha a extrair texto
    - TextPreprocessor falha na limpeza
    - ResumeFeatureExtractor falha na extração
    - ResumeScorer falha na computação
    
    Não expor detalhes técnicos ao cliente. Logar internamente.
    """
  
  def __init__(self, stage: str, candidate_name: Optional[str] = None):
    
    if candidate_name:
      detail = f"Pipeline failed at {stage} for candidate {candidate_name}"
    else:
      detail = f"Pipeline failed at {stage}"

    super().__init__(detail=detail, error_code="PIPELINE_EXECUTION_ERROR")

class SpacyModelError(InternalServerError):
  """spaCy model não carregou corretamente."""

  def __init__(self):
    detail = "NLP model is unavailable. Please contact support."
    super().__init__(detail=detail, error_code="SPACY_MODEL_ERROR")

class DatabaseConnectionError(InternalServerError):
  """Falha de conexão com o banco de dados."""

  def __init__(self):
    detail = "Database connection failed. Please try again later."
    super().__init__(detail=detail, error_code="DATABASE_CONNECTION_ERROR")
