import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# ========================================================================
# Context variables -> Thread-safe storage de request_id
# ========================================================================

request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

def get_request_id() -> str:
  """Obter request_id do contexto atual.
    
    Se não houver request_id definido (ex: código executado fora de request),
    retorna 'no-request-id'.
    
    Returns:
        str: UUID do request ou 'no-request-id'
    """
  
  request_id = request_id_context.get()
  return request_id if request_id else "no-request-id"

def set_request_id(request_id: str) -> None:
  """Definir request_id no contexto atual.
    
    Chamado automaticamente por RequestIDMiddleware.
    Args:
        request_id (str): UUID do request
  """
  request_id_context.set(request_id)

# ========================================================================
# Formatter customizado para incluir request_id em cada log
# ========================================================================

class RequestIDFormatter(logging.Formatter):
  """Formatter customizado que inclui request_id em cada log entry.
    
    Formato: TIMESTAMP | LEVEL | REQUEST_ID | MODULE | MESSAGE
    
    Exemplo de output:
        2026-04-28 10:30:45,123 | INFO | abc-123-def | api.routes.processes | Process created successfully
        2026-04-28 10:30:46,456 | WARNING | abc-123-def | api.services.candidate | CV too short: 42 chars
    
    O request_id vem de ContextVar, portanto é thread-safe e específico
    do contexto de execução atual.
    """
  
  def format(self, record: logging.LogRecord) -> str:
    """Formatar log record com request_id.
        
        Args:
            record: LogRecord do logging module
            
        Returns:
            str: Linha formatada pronta para output
        """
    
    # Obter timestamp em formato ISO-like
    timestamp = datetime.fromtimestamp(record.created).strftime(
      "%Y-%m-%d %H:%M:%S,%f"
    )[:-3]  # Remove os últimos 3 dígitos dos microsegundos para ter milissegundos
    
    # Obter request_id do contexto
    request_id = get_request_id()

    # Obter nome do módulo (ex: api.services.candidate)
    module_name = record.name

    # Obter level name (ex: INFO, WARNING, ERROR)
    level_name = record.levelname

    # Mensagem original
    message = record.getMessage()

    # Se há exceção, incluir traceback
    if record.exc_info:
      exc_text = self.formatException(record.exc_info)
      message = f"{message}\n{exc_text}"

    # Formatar linha final: TIMESTAMP | LEVEL | REQUEST_ID | MODULE | MESSAGE
    return (
      f"{timestamp} | {level_name} | {request_id} | {module_name} | {message}"
    )
  
# ========================================================================
# Middleware -> injeta request_id em cada request
# ========================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
  """Middleware que gera e injeta request_id em cada HTTP request.
    
    Flow:
    1. Request chega em FastAPI
    2. Middleware gera UUID novo (request_id)
    3. Armazena em ContextVar (thread-safe)
    4. Request é processado (loggers acessam request_id via ContextVar)
    5. Response é retornada
    6. Request_id fica disponível em logs de toda a cadeia
    
    Uso em main.py:
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)
    
    Então em qualquer logger:
        logger = logging.getLogger(__name__)
        logger.info("Processing request")  # Automaticamente inclui request_id
    """

  async def dispatch(
      self, 
      request: Request, 
      call_next
  ) -> Response:
    """Processar request com request_id.
        
        Args:
            request: Request do Starlette
            call_next: Callable para continuar pipeline
            
        Returns:
            Response: Response do endpoint
        """
    
    # Gerar UUID novo para request_id
    request_id = str(uuid.uuid4())

    # Armazenar em ContextVar para acesso em loggers
    set_request_id(request_id)

    # Adiciona ao header de resposta
    # Permite que cliente veja qual request_id foi associado a essa requisição
    request.state.request_id = request_id  

    # Log de entrada da request
    logger = logging.getLogger(__name__)
    logger.info(
      f"{request.method} {request.url.path} | Client: {request.client.host if request.client else 'unknown'}"
    )

    try:
      # Chamar próximo middleware ou endpoint
      response = await call_next(request)
    except Exception as e:
      # Log de erro com request_id
      logger.error(
          f"Unhandled exception in request: {str(e)}",
          exc_info=True
      )
      raise # Re-raise para tratamento normal de erros

    # Adicionar request_id no header de resposta
    response.headers["X-Request-ID"] = request_id

    # Log de saída da request
    logger.info(
      f"{request.method} {request.url.path} | Status: {response.status_code}"
    )

    return response
  
# =========================================================================
# Configuração global de logging
# =========================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
  """Configurar logging para toda a aplicação.
    
    Deve ser chamado uma única vez no startup da aplicação (main.py).
    
    Configura:
    1. Root logger com level especificado
    2. Handler para stderr (console)
    3. Handler para ficheiro (optional)
    4. Formatter customizado com request_id
    
    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Caminho para ficheiro de log (optional). Se não especificado,
                  apenas stderr é usado.
    
    Exemplo de uso em main.py:
        from backend.api.utils.logging import setup_logging
        
        if __name__ == "__main__":
            setup_logging(level="INFO", log_file="logs/app.log")
            uvicorn.run(app, host="0.0.0.0", port=8000)
    
    Regra importante: esta função deve ser chamada ANTES de criar qualquer logger.
    Loggers obtêm configuração do root logger, portanto setup_logging deve ser first.
    """

  # Validar level
  numeric_level = getattr(logging, level.upper(), None)
  if not isinstance(numeric_level, int):
      raise ValueError(f"Invalid log level: {level}")
  
  # Obter root logger
  root_logger = logging.getLogger()
  root_logger.setLevel(numeric_level)

  # Remove handlers antigos (caso setup_logging seja chamado mais de uma vez)
  root_logger.handlers.clear()

  # Criar formatter customizado
  formatter = RequestIDFormatter()

  # ========================================================
  # Handler 1: Console (stderr)
  # ========================================================

  console_handler = logging.StreamHandler(sys.stderr)
  console_handler.setLevel(numeric_level)
  console_handler.setFormatter(formatter)
  root_logger.addHandler(console_handler)

  # ========================================================
  # Handler 2: File (opcional)
  # ========================================================

  if log_file:
    try:

      # Criar diretório de logs se não existir
      import os 
      log_dir = os.path.dirname(log_file)
      if log_dir and not os.path.exists(log_dir):
          os.makedirs(log_dir, exist_ok=True)

      file_handler = logging.FileHandler(log_file)
      file_handler.setLevel(numeric_level)
      file_handler.setFormatter(formatter)
      root_logger.addHandler(file_handler)
    except IOError as e:
      root_logger.warning(
        f"Could not create log {log_file}: {str(e)}."
      )

  # ========================================================================
  # Silenciar loggers barulhentos
  # ========================================================================

  # uvicorn logs são muito verbosos em DEBUG
  logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
  logging.getLogger("uvicorn").setLevel(logging.WARNING)

  # SQLAlchemy logs podem ser verbosos em DEBUG, especialmente com echo=True
  logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
  logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

# ========================================================================
# Helpers -> Funções de conveniência para logging
# ========================================================================

def get_logger(name: str) -> logging.Logger:
  """Obter logger nomeado (convenience function).
    
    Equivalente a logging.getLogger(name) mas é mais explícito.
    
    Uso:
        from backend.api.utils.logging import get_logger
        
        logger = get_logger(__name__)
        logger.info("Something happened")
    
    Args:
        name: Nome do logger (tipicamente __name__)
        
    Returns:
        logging.Logger: Logger configurado
    """
  return logging.getLogger(name)

# ============================================================================
# REGRAS DE LOGGING — Documentadas para consistência
# ============================================================================
 
"""
REGRAS DE LOGGING
 
1. NUNCA usar print()
   ✗ print("Starting process")
   ✓ logger.info("Starting process")
 
2. DEBUG — Informação técnica detalhada (dev only)
   logger.debug(f"Processing file {filename}, size={file_size_bytes}")
   logger.debug("Entering ResumeParser.parse()")
   
3. INFO — Eventos significativos (state changes)
   logger.info(f"Process {process_id} created")
   logger.info(f"Candidate {candidate_name} uploaded successfully")
   logger.info("Screening pipeline started")
   
4. WARNING — Situações inesperadas mas recuperáveis
   logger.warning(f"CV parse failed for {filename}: text too short (42 chars)")
   logger.warning(f"Candidate {name} has no education level found, using default")
   
5. ERROR — Operação falhou
   logger.error(f"Failed to write result to database: {str(e)}", exc_info=True)
   logger.error(f"Pipeline failed for candidate {name}: {str(e)}", exc_info=True)
 
NUNCA LOGAR:
   ✗ Raw CV text (confidencial, muito grande)
   ✗ Emails ou phone numbers (PII)
   ✗ Passwords ou tokens (segurança)
   ✗ File contents (muito grande)
 
LOGAR SEMPRE:
   ✓ Nomes de ficheiro (sanitizados/generic)
   ✓ Process IDs, Candidate IDs (UUIDs, não PII)
   ✓ Timestamps
   ✓ Nomes de operações (upload, parse, score)
   ✓ Erros e exceções (com traceback em ERROR)
 
EXEMPLO DE LOG BOM:
 
    logger.info(f"Candidate alice uploaded: file=cv_001.pdf, size=245KB, process={process_id}")
    
    Contém: nome genérico, tamanho (não conteúdo), IDs
    NÃO contém: email, phone, raw text
 
EXEMPLO DE LOG RUIM:
 
    logger.info(f"Candidate alice uploaded: email=alice@company.com, phone=+351912345678, text={raw_cv_text[:500]}")
    
    Problema: PII (email, phone) e CV text (confidencial)
"""