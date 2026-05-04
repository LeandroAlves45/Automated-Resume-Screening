"""Esquemas Pydantic para pedidos, respostas, enumerações e erros da API."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

# ========== Enumerações: estado e papéis ==========


class ProcessStatus(str, Enum):
    """
    Estados possíveis de um processo de screening.

    Ordem esperada: created -> files_uploaded -> processing -> completed | failed
    """

    CREATED = "created"
    FILES_UPLOADED = "files_uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "canceled"


class ParseStatus(str, Enum):
    """Estados da extração de texto de um CV."""

    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


class UserRole(str, Enum):
    """Papéis possíveis de um utilizador."""

    RECRUITER = "recruiter"
    ADMIN = "admin"


class MatchCategory(str, Enum):
    """Classificação do candidato com base na pontuação total."""

    STRONG_MATCH = "strong_match"
    POTENTIAL_MATCH = "potential_match"
    WEAK_MATCH = "weak_match"


# ======================================================================
# Modelos de pedido — entrada do utilizador
# ======================================================================


class ProcessCreate(BaseModel):
    """
    Contrato para criar um novo processo de screening.

    Validações:
    - title: obrigatório, máximo 255 caracteres
    - jd_text: obrigatório, mínimo 50 caracteres
    """

    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Nome descritivo do processo de screening",
        example="Backend Developer — Porto",
    )

    jd_text: str = Field(
        ...,
        min_length=50,
        description="Texto integral da descrição da vaga",
        example="We are looking for a Python developer with 5+ years experience in backend development, FastAPI, and PostgreSQL.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Backend Developer — Porto",
                "jd_text": "We are looking for a Python developer with 5+ years experience in backend development, FastAPI, and PostgreSQL.",
            }
        }
    )


class UserRegister(BaseModel):
    """Contrato para registo de um utilizador."""

    email: EmailStr = Field(
        ...,
        description="Endereço de correio eletrónico (formato válido)",
    )

    password: str = Field(
        ...,
        min_length=8,
        description="Palavra-passe com pelo menos 8 caracteres",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "recruiter@company.com",
                "password": "strongpassword123",
            }
        }
    )


class LoginRequest(BaseModel):
    """Contrato para início de sessão do utilizador."""

    email: EmailStr = Field(
        ...,
        description="Endereço de correio eletrónico (formato válido)",
    )

    password: str = Field(
        ...,
        min_length=8,
        description="Palavra-passe com pelo menos 8 caracteres",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "recruiter@company.com",
                "password": "strongpassword123",
            }
        }
    )


# ======================================================================
# Modelos de resposta — saída da API
# ======================================================================


class ProcessResponse(BaseModel):
    """
    Representação de um processo de screening nas respostas da API.

    Corresponde ao ORM Process com lógica de serialização.
    - completed_at: None até o processo estar concluído
    - error_message: None se não houve falha
    """

    process_id: UUID = Field(
        ..., description="Identificador único do processo de screening"
    )
    title: str = Field(..., description="Nome descritivo do processo de screening")
    status: ProcessStatus = Field(
        ..., description="Estado atual do processo de screening"
    )
    created_at: datetime = Field(..., description="Data/hora de criação do processo")
    updated_at: datetime = Field(
        ..., description="Data/hora da última atualização do processo"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Data/hora de conclusão; null se ainda não terminou",
    )
    error_message: Optional[str] = Field(
        None,
        description="Detalhe do erro se o processo falhou; null caso contrário",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "process_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Backend Developer — Porto",
                "status": "completed",
                "created_at": "2026-04-28T10:30:00+00:00",
                "updated_at": "2026-04-28T10:35:00+00:00",
                "completed_at": "2026-04-28T10:35:00+00:00",
                "error_message": None,
            }
        }
    )


class ProcessListResponse(BaseModel):
    """Resposta de listagem de processos."""

    processes: list[ProcessResponse] = Field(
        default_factory=list,
        description="Lista de processos, ordenada por created_at descendente",
    )


class UploadFailure(BaseModel):
    """Informação sobre um ficheiro que falhou no processamento."""

    filename: str = Field(
        ...,
        description="Nome original do ficheiro enviado que falhou",
    )

    reason: str = Field(
        ...,
        description="Motivo da falha, p.ex. 'ficheiro demasiado grande', 'formato não suportado'",
    )


class UploadResponse(BaseModel):
    """Resposta após o envio de ficheiros.

    Regra: mesmo que alguns ficheiros falhem, o upload responde 200 OK.
    Ficheiros com sucesso ficam registados em Candidate.
    Ficheiros com falha aparecem em 'failed' sem abortar o lote.
    """

    process_id: UUID = Field(
        ...,
        description="ID do processo de screening associado ao upload",
    )
    uploaded: int = Field(
        ...,
        ge=0,
        description="Número de ficheiros enviados e processados com sucesso",
    )
    failed: list[UploadFailure] = Field(
        default_factory=list,
        description="Lista de ficheiros que falharam com respetivos motivos",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "process_id": "550e8400-e29b-41d4-a716-446655440000",
                "uploaded": 3,
                "failed": [
                    {
                        "filename": "resume_john_doe.pdf",
                        "reason": "file too short, less than 20 words",
                    },
                ],
            }
        }
    )


class ScoreBreakdown(BaseModel):
    """
    Desagregação da pontuação por critério.

    Cada valor em [0.0, 100.0]. A soma ponderada corresponde a total_score.
    Valores brutos antes de aplicar pesos (auditoria).
    """

    skills_match: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Pontuação bruta para competências (0–100)",
    )
    experience_years: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Pontuação bruta para experiência (0–100)",
    )
    education: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Pontuação bruta para formação (0–100)",
    )
    keyword_density: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Pontuação bruta para densidade de palavras-chave (0–100)",
    )


class ResultResponse(BaseModel):
    """Pontuação e detalhe de um candidato.

    Corresponde ao ORM Result com dados do Candidate:
    - matched_skills e missing_skills são listas de strings
    - breakdown segue o schema ScoreBreakdown
    - category é STRONG/POTENTIAL/WEAK consoante total_score
    """

    rank: int = Field(
        ...,
        ge=1,
        description="Posição do candidato no processo (1 = melhor correspondência)",
    )

    name: str = Field(
        ...,
        description="Nome extraído do CV, ou nome do ficheiro se não existir nome",
    )

    total_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Pontuação global (soma ponderada dos critérios, 0–100)",
    )

    category: MatchCategory = Field(
        ...,
        description="Categoria consoante limiares (strong_match, potential_match, weak_match)",
    )

    breakdown: ScoreBreakdown = Field(
        ...,
        description="Detalhe que corresponde exatamente ao schema ScoreBreakdown.",
    )

    matched_skills: list[str] = Field(
        default_factory=list,
        description="Competências da JD encontradas no CV",
    )

    required_skills: list[str] = Field(
        default_factory=list,
        description="Todas as competências exigidas na JD",
    )

    missing_skills: list[str] = Field(
        default_factory=list,
        description="Competências da JD não encontradas no CV",
    )

    experience_years_found: int = Field(
        ...,
        ge=0,
        description="Anos de experiência identificados no CV",
    )

    @model_validator(mode="before")
    @classmethod
    def calculate_missing_skills(cls, data: Any) -> Any:
        """Calcula missing_skills como diferença de conjuntos (required - matched)."""

        if not isinstance(data, dict):
            return data
        if "required_skills" in data and "matched_skills" in data:
            updated = dict(data)
            required = set(data["required_skills"])
            matched = set(data["matched_skills"])
            updated["missing_skills"] = sorted(required - matched)
            return updated
        return data

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rank": 1,
                "name": "alice",
                "total_score": 82.5,
                "category": "strong_match",
                "breakdown": {
                    "skills_match": 85.0,
                    "experience_years": 90.0,
                    "education": 75.0,
                    "keyword_density": 78.0,
                },
                "matched_skills": ["python", "docker", "postgresql"],
                "required_skills": ["python", "docker", "postgresql", "kubernetes"],
                "missing_skills": ["kubernetes"],
                "experience_years_found": 6,
            }
        }
    )


class ResultsSummary(BaseModel):
    """Estatísticas agregadas dos resultados."""

    total: int = Field(..., ge=0, description="Número total de candidatos processados")
    strong_matches: int = Field(..., ge=0, description="Contagem de strong matches")
    potential_matches: int = Field(
        ..., ge=0, description="Contagem de potential matches"
    )
    weak_matches: int = Field(..., ge=0, description="Contagem de weak matches")


class ResultsResponse(BaseModel):
    """
    Resposta com todos os resultados do screening.

    Devolvida quando o estado é 'completed'.
    Se o estado é 'processing', a API pode responder 202 apenas com {'status': 'processing'}.
    """

    process_id: UUID = Field(..., description="ID do processo de screening")
    status: ProcessStatus = Field(
        ..., description="Estado atual do processo de screening"
    )
    summary: ResultsSummary = Field(
        ..., description="Estatísticas agregadas dos resultados"
    )
    candidates: list[ResultResponse] = Field(
        default_factory=list,
        description="Candidatos com pontuações e detalhes, ordenados por rank",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "process_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "summary": {
                    "total": 10,
                    "strong_matches": 3,
                    "potential_matches": 4,
                    "weak_matches": 3,
                },
                "candidates": [
                    {
                        "rank": 1,
                        "name": "alice",
                        "total_score": 82.5,
                        "category": "strong_match",
                        "breakdown": {
                            "skills_match": 85.0,
                            "experience_years": 90.0,
                            "education": 75.0,
                            "keyword_density": 78.0,
                        },
                        "matched_skills": ["python", "docker"],
                        "required_skills": ["python", "docker", "kubernetes"],
                        "missing_skills": ["kubernetes"],
                        "experience_years_found": 6,
                    }
                ],
            }
        }
    )


class UserResponse(BaseModel):
    """Informação pública do utilizador."""

    user_id: UUID = Field(..., description="Identificador único do utilizador")
    email: EmailStr = Field(..., description="Endereço de correio eletrónico")
    role: UserRole = Field(..., description="Papel (recruiter ou admin)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "recruiter@company.com",
                "role": "recruiter",
            }
        }
    )


class CandidateResponse(BaseModel):
    """Informação de um candidato após o upload."""

    id: UUID = Field(..., description="Identificador único do candidato")
    name: str = Field(..., description="Nome extraído do CV ou do ficheiro")
    original_filename: str = Field(..., description="Nome original do CV enviado")
    parse_status: ParseStatus = Field(
        ...,
        description="Estado da extração de texto a partir do CV",
    )


class TokenResponse(BaseModel):
    """Resposta de autenticação com token JWT."""

    access_token: str = Field(
        ...,
        description="Token JWT Bearer",
    )
    token_type: str = Field(
        default="bearer",
        description="Tipo do token; normalmente 'bearer'",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxJV_adQssw5c...",
                "token_type": "bearer",
            }
        }
    )

class RefreshTokenRequest(BaseModel):
    """
    Contrato para requisição de refresh de access token.
    
    Endpoint: POST /api/auth/refresh
    
    O refresh_token foi recebido em login() e deve ser enviado aqui
    para obter um novo access_token sem precisar de credenciais novamente.
    """

    refresh_token: str = Field(
        ...,
        description="Refresh token received in login endpoint",
    )

    model_config = ConfigDict(json_schema_extra={
      "example": {
        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxJV_adQssw5c...",
      }
    })

class RefreshTokenResponse(BaseModel):
    """
    Resposta de refresh de token.
    
    Retorna um novo access_token válido por 1h.
    O refresh_token permanece o mesmo e continua válido por 7 dias.
    """

    access_token: str = Field(..., description="New access token JWT valid for 1 hour")

    token_type: str = Field(default="bearer", description="Type of token; always 'bearer'")

    model_config = ConfigDict(json_schema_extra={
      "example": {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxJV_adQssw5c...",
        "token_type": "bearer",
      }
    })

class LogoutRequest(BaseModel):
    """
    Contrato para logout de utilizador.
    
    Endpoint: POST /api/auth/logout
    
    Ambos tokens são revogados imediatamente:
    - access_token é adicionado a TokenBlacklist
    - refresh_token é marcado como is_revoked=True
    
    refresh_token é opcional: se não fornecido, apenas access_token é revogado.
    """

    refresh_token: str | None = Field(
      None,
      description="Refresh token to revoke (optional, but recommended for complete logout)",
    )

    model_config = ConfigDict(json_schema_extra={
      "example": {
        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxJV_adQssw5c...",
      }
    })

class HealthResponse(BaseModel):
    """
    Resposta do health check — indica se o sistema está operacional.

    GET /health não exige autenticação e deve responder em < 500 ms.
    Verifica: API em execução, base de dados ligada, modelo spaCy carregado.
    """

    status: str = Field(
        ...,
        description="Estado geral; esperado 'ok' se todas as verificações passarem",
    )
    version: str = Field(..., description="Versão atual da API")
    database: str = Field(
        ...,
        description="Estado da ligação à base de dados: 'connected' ou 'disconnected'",
    )
    nlp_model: str = Field(
        ...,
        description="Estado do modelo NLP: 'loaded' ou 'not loaded'",
    )
    environment: str = Field(
        ...,
        description="Ambiente da aplicação (development, production, …)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "version": "2.0.0",
                "database": "connected",
                "nlp_model": "loaded",
                "environment": "development",
            }
        }
    )


# =============================================================
# Modelos de erro — respostas de erro da API
# =============================================================


class ErrorDetail(BaseModel):
    """
    Detalhe de um erro devolvido pela API.

    Usado pelas rotas para erros em JSON estruturado.
    """

    detail: str = Field(
        ...,
        description="Mensagem de erro (validação ou outra falha)",
    )
    error_code: Optional[str] = Field(
        None,
        description="Código que identifica o tipo de erro",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Process not found",
                "error_code": "PROCESS_NOT_FOUND",
            }
        }
    )


# =============================================================
# Modelos auxiliares — operações assíncronas
# =============================================================


class ProcessingStatusResponse(BaseModel):
    """
    Resposta para consultar o estado de um processo assíncrono.

    Usada em endpoints que devolvem 202 Accepted e polling.
    Indica se o processo está 'processing' ou já 'completed' / 'failed'.
    """

    status: ProcessStatus = Field(..., description="Estado, p.ex. 'processing'")
    message: str = Field(
        default="Process is still running. Please check back later for results.",
        description="Mensagem informativa sobre o processamento",
    )
