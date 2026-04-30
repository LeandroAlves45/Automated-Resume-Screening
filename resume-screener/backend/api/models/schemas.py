from datetime import datetime
from enum import Enum
from turtle import mode
from typing import Optional
from uuid import UUID

from numpy import require
from pydantic import BaseModel, Field, EmailStr, ConfigDict, model_validator

# ========== Enumerações: Estado e Roles =========

class ProcessStatus(str, Enum):
  """
  Estados possíveis de um screening process.
  
  Ordem esperada: created -> files_uploaded -> processing -> completed | failed 
  """

  CREATED = "created"
  FILES_UPLOADED = "files_uploaded"
  PROCESSING = "processing"
  COMPLETED = "completed"
  FAILED = "failed"
  CANCELLED = "canceled"

class ParseStatus(str, Enum):
  """Estados da extração de um texto de um CV."""
  PENDING = "pending"
  SUCCESS = "success"
  ERROR = "error"


class UserRole(str, Enum):
  """Papéis possíveis de um usuário."""
  RECRUITER = "recruiter"
  ADMIN = "admin"

class MatchCategory(str, Enum):
  """Classificação de candidato baseada no score total."""
  STRONG_MATCH = "strong_match"
  POTENTIAL_MATCH = "potential_match"
  WEAK_MATCH = "weak_match"

# ======================================================================
# Modelos de request -> Entrada do usuário
# ======================================================================

class ProcessCreate(BaseModel):
  """
  Contrato para criação de um novo processo de screening.

  Validações:
  - title: obrigatório, máximo 255 caracteres
  - jd_text: obrigatório, mínimo 50 caracteres
  """

  title: str = Field(
    ...,
    min_length=1,
    max_length=255,
    description="Descriptive name of the screening process"
  )

  jd_text: str = Field(
    ...,
    min_length=50,
    description="Full text of the job description"
  )

  model_config = ConfigDict(json_schema_extra={
    "example": {
      "title": "Backend Developer Porto",
      "jd_text": "Seeking senior Python/FastAPI developer with 3+ years backend experience, NLP expertise, and proven ability to build scalable systems."
    }
  })

class UserRegister(BaseModel):
  """Contrato para registar um usuário."""
  email: EmailStr = Field(
    ...,
    description="User's email address, must be valid format"
  )

  password: str = Field(
    ...,
    min_length=8,
    description="Password with minimum 8 characters"
  )

  model_config = ConfigDict(json_schema_extra={
    "example": {
      "email": "recruiter@company.com",
      "password": "strongpassword123"
    }
  })

class LoginRequest(BaseModel):
  """Contrato para login de usuário."""
  email: EmailStr = Field(
    ...,
    description="User's email address, must be valid format"
  )

  password: str = Field(
    ...,
    min_length=8,
    description="Password with minimum 8 characters"
  )

  model_config = ConfigDict(json_schema_extra={
    "example": {
      "email": "recruiter@company.com",
      "password": "strongpassword123"
    }
  })

# ======================================================================
# Modelos de response -> Saída da API
# ======================================================================

class ProcessResponse(BaseModel):
  """
  Representação de um screening process para respostas da API.

  Mapeia para ORM Process mas adiciona lógica de serialização.
  - completed_at: None até o processo ser finalizado
  - error_message: None se não houve falha
  """

  process_id: UUID = Field(..., description="Unique identifier of the screening process")
  title: str = Field(..., description="Descriptive name of the screening process")
  status: ProcessStatus = Field(..., description="Current status of the screening process")
  created_at: datetime = Field(..., description="Timestamp when the process was created") 
  updated_at: datetime = Field(..., description="Timestamp of the last update to the process")
  completed_at: Optional[datetime] = Field(
    None, 
    description="Timestamp when the process was completed, null if not finished"
  )
  error_message: Optional[str] = Field(
    None, 
    description="Error details if the process failed, null otherwise"
  )

  model_config = ConfigDict(json_schema_extra={
    "example": {
      "process_id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Backend Developer — Porto",
      "status": "completed",
      "created_at": "2026-04-28T10:30:00+00:00",
      "updated_at": "2026-04-28T10:35:00+00:00",
      "completed_at": "2026-04-28T10:35:00+00:00",
      "error_message": None
    }
  })

class ProcessListResponse(BaseModel):
  """Resposta de listagem de processos"""
  
  processes: list[ProcessResponse] = Field(
    default_factory=list,
    description="List of screening processes ordered by created_at descending"
  )

class UploadFailure(BaseModel):
  """Informação de um ficheiro que falhou ao ser processado."""

  filename: str = Field(
    ...,
    description="Original filename of the uploaded file that failed processing"
  )

  reason: str = Field(
    ...,
    description="Reason for the failure, e.g., 'file too large', 'unsupported format'"
  )

class UploadResponse(BaseModel):
  """Resposta após upload de ficheiros.
    
    Regra: mesmo que alguns ficheiros falhem, upload é 200 OK.
    Ficheiros com sucesso são registados em Candidate.
    Ficheiros com falha são listados em 'failed' sem abort do batch.
    """
  
  process_id: UUID = Field(
    ..., 
    description="ID of the screening process associated with the upload"
  )
  uploaded: int = Field(
    ...,
    ge=0, 
    description="Number of files successfully uploaded and processed"
  )
  failed: list[UploadFailure] = Field(
    default_factory=list,
    description="List of files that failed to upload with reasons"
  )


  model_config = ConfigDict(json_schema_extra={
    "example": {
      "process_id": "550e8400-e29b-41d4-a716-446655440000",
      "uploaded": 3,
      "failed": [
        {
          "filename": "resume_john_doe.pdf",
          "reason": "file too short, less than 20 words"
        },
      ]
    }
  })

class ScoreBreakdown(BaseModel):
  """
    Desagregação do score por critério individual.
    
    Cada valor em [0.0, 100.0]. Soma ponderada = total_score.
    Inclui ANTES de aplicar pesos para auditoria.
    """
  
  skills_match: float = Field(
    ...,
    ge=0.0, 
    le=100.0, 
    description="Raw match score for skills criteria (0-100)"
  )
  experience_years: float = Field(
    ...,
    ge=0.0, 
    le=100.0, 
    description="Raw match score for experience criteria (0-100)"
  )
  education: float = Field(
    ...,
    ge=0.0, 
    le=100.0,
    description="Raw match score for education criteria (0-100)"
  )
  keyword_density: float = Field(
    ...,
    ge=0.0, 
    le=100.0,
    description="Raw match score for keyword density criteria (0-100)"
  )

class ResultResponse(BaseModel):
  """Score e breakdown de um candidato.
    
    Mapeia para ORM Result mas com informação do Candidate:
    - matched_skills e missing_skills são listas de strings
    - breakdown é um dict com os 4 critérios
    - category é STRONG/POTENTIAL/WEAK baseado no total_score
    """

  rank: int = Field(
    ...,
    ge=1,
    description="Rank of the candidate within the screening process, starting at 1 for the best match"
  )

  name: str = Field(
    ...,
    description="Extracted name of the candidate from the resume, or filename if name not found"
  )

  total_score: float = Field(
    ...,
    ge=0.0,
    le=100.0,
    description="Overall match score for the candidate, calculated as a weighted sum of criteria (0-100)"
  )

  category: MatchCategory = Field(
    ...,
    description="Match category based on total_score thresholds (strong_match, potential_match, weak_match)"
  )

  breakdown: ScoreBreakdown = Field(
    ...,
    description="Detailed breakdown matching ScoreBreakdown schema exactly."
  )

  matched_skills: list[str] = Field(
    default_factory=list,
    description="List of skills from the JD that were found in the candidate's resume"
  )

  required_skills: list[str] = Field(
    default_factory=list,
    description="List of all skills from the JD that were required for the position"
  )

  missing_skills: list[str] = Field(
    default_factory=list,
    description="List of skills from the JD that were NOT found in the candidate's resume"
  )

  experience_years_found: int = Field(
    ...,
    ge=0,
    description="Number of years of experience found in the candidate's resume"
  )

  @model_validator(mode="before")
  def calculate_missing_skills(self):
    """Calcula missing_skills como required_skills - matched_skills."""

    if 'required_skills' in self and 'matched_skills' in self:
      required = set(self['required_skills'])
      matched = set(self['matched_skills'])
      self['missing_skills'] = sorted(required - matched)
    return self
  
  model_config = ConfigDict(json_schema_extra={
    "example": {
        "rank": 1,
        "name": "alice",
        "total_score": 82.5,
        "category": "strong_match",
        "breakdown": {
            "skills_match": 85.0,
            "experience_years": 90.0,
            "education": 75.0,
            "keyword_density": 78.0
        },
        "matched_skills": ["python", "docker", "postgresql"],
        "required_skills": ["python", "docker", "postgresql", "kubernetes"],
        "missing_skills": ["kubernetes"],
        "experience_years_found": 6
      }
  })

class ResultsSummary(BaseModel):
  """Estatísticas agregadas dos resultados"""
  total: int = Field(..., ge=0, description="Total number of candidates processed")
  strong_matches: int = Field(..., ge=0, description="Count of Strong Matches")
  potential_matches: int = Field(..., ge=0, description="Count of Potential Matches")
  weak_matches: int = Field(..., ge=0, description="Count of Weak Matches")

class ResultsResponse(BaseModel):
  """
    Resposta com todos os resultados de screening.
    
    Regra: returned quando status é 'completed'.
    Se status é 'processing', retorna 202 Accepted com apenas {'status': 'processing'}.
    """
  process_id: UUID = Field(..., description="ID of the screening process")
  status: ProcessStatus = Field(..., description="Current status of the screening process")
  summary: ResultsSummary = Field(..., description="Aggregated statistics of the results")
  candidates: list[ResultResponse] = Field(
    default_factory=list,
    description="List of candidates with their scores and breakdowns, ordered by rank"
  )

  model_config = ConfigDict(json_schema_extra={
        "example": {
            "process_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "summary": {
                "total": 10,
                "strong_matches": 3,
                "potential_matches": 4,
                "weak_matches": 3
            },
            "candidates": [
                {
                    "rank": 1,
                    "name": "alice",
                    "total_score": 82.5,
                    "category": "Strong Match",
                    "breakdown": {
                        "skills_match": 85.0,
                        "experience_years": 90.0,
                        "education": 75.0,
                        "keyword_density": 78.0
                    },
                    "matched_skills": ["python", "docker"],
                    "required_skills": ["python", "docker", "kubernetes"],
                    "missing_skills": ["kubernetes"],
                    "experience_years_found": 6
                }
            ]
        }
    })
  
class UserResponse(BaseModel):
  """Informações públicas do usuário"""
  user_id: UUID = Field(..., description="Unique identifier of the user")
  email: EmailStr = Field(..., description="Email address of the user")
  role: UserRole = Field(..., description="Role of the user (recruiter or admin)")
  
  model_config = ConfigDict(json_schema_extra={
    "example": {
      "user_id": "123e4567-e89b-12d3-a456-426614174000",
      "email": "recruiter@company.com",
      "role": "recruiter"
    }
  })

class CandidateResponse(BaseModel):
  """Informações de um candidato após upload."""
  id: UUID = Field(..., description="Unique identifier of the candidate")
  name: str = Field(..., description="Extracted name from the resume or filename")
  original_filename: str = Field(..., description="Original filename of the uploaded resume")
  parse_status: ParseStatus = Field(..., description="Status of the text extraction from the resume")

class TokenResponse(BaseModel):
  """Resposta de autenticação com JWT Token."""

  access_token: str = Field(
    ...,
    description="JWT Bearer Token"
  )
  token_type: str = Field(
    default="bearer",
    description="Type of the token, allways 'bearer'"
  )

  model_config = ConfigDict(json_schema_extra={
    "example": {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxJV_adQssw5c...",
      "token_type": "bearer"
    }
  })

class HealthResponse(BaseModel):
  """
    Resposta de health check — diagnosticar se sistema está operacional.
    
    GET /health não requer autenticação e deve responder < 500ms.
    Verifica: API running, database connected, spaCy model loaded.
    """
  
  status: str = Field(..., description="Overall health status, expected 'ok' if all checks pass")
  version: str = Field(..., description="Current version of the API")
  database: str = Field(..., description="Status of database connection, 'connected' or 'disconnected'")
  nlp_model: str = Field(..., description="Status of NLP model, 'loaded' or 'not loaded'")
  environment: str = Field(..., description="Current application environment (development, production)")

  model_config = ConfigDict(json_schema_extra={
    "example": {
      "status": "ok",
      "version": "2.0.0",
      "database": "connected",
      "nlp_model": "loaded",
      "environment": "development"
    }
  })

# =============================================================
# Modelos de erro -> Respostas de erro da API
# =============================================================

class ErrorDetail(BaseModel):
  """
    Detalhe de um erro retornado pela API.
    
    Usado por rotas para retornar erros em JSON estruturado.
    """
  detail: str = Field(
    ..., 
    description="Detailed error message describing the validation issue or reason for failure"
  )
  error_code: Optional[str] = Field(
    None, 
    description="Error code identifying the type of error"
  )
  
  model_config = ConfigDict(json_schema_extra={
    "example": {
      "detail": "Process not found",
      "error_code": "PROCESS_NOT_FOUND"
    }
  })

# =============================================================
# Modelos auxiliares -> Resposta para operações assíncronas
# =============================================================

class ProcessingStatusResponse(BaseModel):
  """
    Resposta para verificar status de um processo assíncrono.
    
    Usado para endpoints que retornam 202 Accepted e precisam de polling.
    Indica se processo ainda está 'processing' ou já foi 'completed' ou 'failed'.
    """

  status: ProcessStatus = Field(..., description="Status: 'processing'")
  message: str = Field(
    default="Process is still running. Please check back later for results.",
    description="Informational message about the processing status"
  )