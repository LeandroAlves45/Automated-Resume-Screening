import re
import string
from pathlib import Path
from typing import Optional

import magic

from backend.api.scoring_config import SUPPORTED_EXTENSIONS
from backend.api.utils.errors import (
  FileTooLargeError,
  InvalidMimeTypeError,
  UnsupportedFileTypeError,
)

# ========================================================================
# Configuração -> Constantes de validação
# ========================================================================

# Mapeamento: extensão -> MIME types permitidos
MIME_TYPE_MAPPING = {
  "pdf": "application/pdf",
  "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "txt": "text/plain"
}

# Tamanho máximo do arquivo (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ========================================================================
# Validação da extensão -> Whitelist
# ========================================================================

def validate_file_extension(filename: str) -> str:
  """Validar extensão do ficheiro contra whitelist.
    
    Extrai extensão (com ponto) e verifica se está em SUPPORTED_EXTENSIONS.
    Comparação case-insensitive (converte para lowercase).
    
    Args:
        filename: Nome do ficheiro (ex: 'resume.pdf')
        
    Returns:
        str: Extensão em lowercase (ex: '.pdf')
        
    Raises:
        UnsupportedFileTypeError: Se extensão não está na whitelist
        
    Exemplos:
        validate_file_extension("resume.pdf")      → ".pdf"
        validate_file_extension("CV.DOCX")         → ".docx"
        validate_file_extension("data.xlsx")       → UnsupportedFileTypeError
        validate_file_extension("noextension")     → UnsupportedFileTypeError
    """
  
  # Extrai extensão (com ponto)
  extension = Path(filename).suffix.lower()  # Ex: '.pdf'

  # Validar
  if not extension or extension not in SUPPORTED_EXTENSIONS:
    extensions_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise UnsupportedFileTypeError(filename, list(SUPPORTED_EXTENSIONS))
  
  return extension

# ========================================================================
# Validação do MIME type -> python-magic
# ========================================================================

def validate_mime_type(
    file_bytes: bytes,
    extension: str,
    filename: str,
) -> str:
  """Validar MIME type via python-magic.
    
    NUNCA confiar em Content-Type header do cliente. Usar python-magic
    para detectar MIME type real baseado no conteúdo do ficheiro (magic bytes).
    
    Validações:
    1. Detecta MIME type real (magic bytes)
    2. Compara com MIME esperado para a extensão
    3. Rejeita se não coincidem (ex: ficheiro .pdf com conteúdo texto)
    
    Args:
        file_bytes: Conteúdo do ficheiro em bytes
        extension: Extensão já validada (ex: '.pdf')
        filename: Nome do ficheiro (para mensagem de erro)
        
    Returns:
        str: MIME type detectado (ex: 'application/pdf')
        
    Raises:
        InvalidMimeTypeError: Se MIME type não corresponde à extensão
        
    Exemplos:
        # Ficheiro legítimo
        with open("resume.pdf", "rb") as f:
            mime = validate_mime_type(f.read(), ".pdf", "resume.pdf")
            → "application/pdf"
        
        # Ficheiro renomeado (conteúdo de TXT com extensão .pdf)
        validate_mime_type(b"hello world", ".pdf", "fake.pdf")
            → InvalidMimeTypeError (porque conteúdo é text/plain, não PDF)
    """
  
  # Detecta MIME type real usando python-magic
  detected_mime = magic.from_buffer(file_bytes, mime=True)  # Ex: 'application/pdf'

  # Obter MIME esperado para a extensão (sem ponto)
  expected_mime = MIME_TYPE_MAPPING.get(extension.lstrip(".")) # Ex: 'application/pdf'

  if not expected_mime:
    # Extensão validada mas não tem mapping (nunca acontece se validate_extension passou)
    raise InvalidMimeTypeError(filename, detected_mime)
  
  # Comparar MIME detectado com esperado
  if extension == ".txt":
    # Para .txt, aceitar qualquer MIME que comece com 'text/'
    if not detected_mime.startswith("text/"):
      raise InvalidMimeTypeError(filename, detected_mime)
    
  else:
    # Para outros tipos, exigir correspondência exata
    if detected_mime != expected_mime:
      raise InvalidMimeTypeError(filename, detected_mime)
    
  return detected_mime

# ========================================================================
# Validação do tamanho do ficheiro
# ========================================================================

def validate_file_size(
    file_bytes: bytes,
    filename: str,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES
) -> None:
  """Validar tamanho do ficheiro contra limite máximo.
    
    Args:
        file_bytes: Conteúdo do ficheiro em bytes
        filename: Nome do ficheiro (para mensagem de erro)
        max_size_bytes: Limite máximo em bytes (default: 10 MB)
        
    Returns:
        None: Se validação passa
        
    Raises:
        FileTooLargeError: Se ficheiro excede limite
        
    Exemplos:
        validate_file_size(b"small", "file.pdf", max_size_bytes=1024)
            → OK
        
        validate_file_size(huge_bytes, "file.pdf", max_size_bytes=1024)
            → FileTooLargeError
    """
  
  file_size_bytes = len(file_bytes)

  if file_size_bytes > max_size_bytes:
    # Converter para MB para mensagem de erro
    max_size_mb = max_size_bytes / (1024 * 1024)
    raise FileTooLargeError(filename, int(max_size_mb))
  

# ========================================================================
# Sanitização de filename -> Segurança contra path traversal
# ========================================================================

def sanitize_filename(filename: str) -> str:
  """Sanitizar nome do ficheiro removendo caracteres perigosos.
    
    Objetivo: prevenir path traversal attacks (ex: ../../etc/passwd).
    
    Remove:
    - Path separators: /, \
    - Null bytes: \0
    - Caracteres de controlo
    - Caracteres não-ASCII (apenas ASCII alphanumeric + . - _)
    
    Resultado: string segura para usar em construção de paths.
    
    Importante: NÃO usar este resultado como identificador único.
    Em CandidateService, usar UUID + filename sanitizado.
    
    Args:
        filename: Nome original (ex: 'my resume.pdf')
        
    Returns:
        str: Filename sanitizado (ex: 'my_resume.pdf')
        
    Exemplos:
        sanitize_filename("resume.pdf")           → "resume.pdf"
        sanitize_filename("my resume.pdf")        → "my_resume.pdf"
        sanitize_filename("../../../etc/passwd")  → "etcpasswd"
        sanitize_filename("file\x00name.pdf")     → "filename.pdf"
        sanitize_filename("café.pdf")             → "caf.pdf"
    """
  
  # Remove null bytes
  filename = filename.replace("\0", "")

  # Remover path separators 
  filename = filename.replace("/", "").replace("\\", "")

  # Permitidos: ASCII alphanumeric, ponto, hífen, underscore
  allowed_chars = set(string.ascii_letters + string.digits + "._-")

  sanitized = ""
  for char in filename:
    if char in allowed_chars:
      sanitized += char
    elif char == " ":
      sanitized += "_"  # Substituir espaços por underscore

  # Remover múltiplos underscores/pontos consecutivos
  sanitized = re.sub(r"_+", "_", sanitized)
  sanitized = re.sub(r"\.+", ".", sanitized)

  # Remover leading/trailing underscores/pontos
  sanitized = sanitized.strip("_.")

  # Se ficou vazio, usar default
  if not sanitized:
    sanitized = "file"

  return sanitized

# ========================================================================
# Validação completa -> Orquestração
# ========================================================================

def validate_upload_file(
    file_bytes: bytes,
    filename: str,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES
) -> dict:
  """Validar ficheiro de upload — orquestra todas as validações.
    
    Pipeline de validação:
    1. Extensão — contra whitelist
    2. MIME type — via magic bytes
    3. Tamanho — contra limite máximo
    4. Sanitização — gerar nome seguro
    
    Esta é a função pública que candidate_service.py deve chamar.
    
    Args:
        file_bytes: Conteúdo completo do ficheiro em bytes
        filename: Nome original do ficheiro (fornecido pelo cliente)
        max_size_bytes: Limite máximo em bytes (configurável)
        
    Returns:
        dict: Resultado da validação:
            {
                "is_valid": True,
                "original_filename": "resume.pdf",
                "sanitized_filename": "resume.pdf",
                "extension": ".pdf",
                "mime_type": "application/pdf",
                "size_bytes": 245000
            }
            
    Raises:
        UnsupportedFileTypeError: Extensão não suportada
        InvalidMimeTypeError: Conteúdo não corresponde à extensão
        FileTooLargeError: Ficheiro muito grande
        
    Exemplos:
        # Ficheiro válido
        result = validate_upload_file(
            file_bytes=pdf_content,
            filename="resume.pdf",
            max_size_bytes=10*1024*1024
        )
        → {
            "is_valid": True,
            "original_filename": "resume.pdf",
            "sanitized_filename": "resume.pdf",
            "extension": ".pdf",
            "mime_type": "application/pdf",
            "size_bytes": 245000
        }
        
        # Ficheiro com extensão não suportada
        result = validate_upload_file(
            file_bytes=xlsx_content,
            filename="data.xlsx"
        )
        → UnsupportedFileTypeError
        
        # Ficheiro renomeado (conteúdo TXT, extensão PDF)
        result = validate_upload_file(
            file_bytes=b"Hello world",
            filename="fake.pdf"
        )
        → InvalidMimeTypeError
    """
  
  # 1. Validar extensão
  extension = validate_file_extension(filename)

  # 2. Validar MIME type
  mime_type = validate_mime_type(file_bytes, extension, filename)

  # 3. Validar tamanho
  validate_file_size(file_bytes, filename, max_size_bytes)

  # 4. Sanitizar filename
  sanitized_filename = sanitize_filename(filename)

  # Tudo passou -> retornar resultado detalhado
  return {
    "is_valid": True,
    "original_filename": filename,
    "sanitized_filename": sanitized_filename,
    "extension": extension,
    "mime_type": mime_type,
    "size_bytes": len(file_bytes)
  }

# ========================================================================
# Helpers -> funções utilitárias para validação
# ========================================================================

def get_supported_extensions_display() -> str:
  """Obter string formatada de extensões suportadas.
    
    Útil para mensagens de erro e documentação.
    
    Returns:
        str: Ex: ".pdf, .docx, .txt"
    """
  return ", ".join(sorted(SUPPORTED_EXTENSIONS))

def get_max_file_size_mb(max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> float:
  """Converter tamanho máximo de bytes para megabytes.
    
    Útil para mensagens de erro e documentação.
    
    Args:
        max_size_bytes: Tamanho máximo em bytes (default: 10 MB)
        
    Returns:
        float: Tamanho máximo em megabytes (ex: 10)
    """
  return max_size_bytes / (1024 * 1024)