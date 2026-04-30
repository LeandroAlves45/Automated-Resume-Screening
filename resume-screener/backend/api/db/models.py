import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
  String,
  Text,
  Float,
  Integer,
  Boolean,
  DateTime,
  ForeignKey,
  JSON,
  Enum,
  text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum 

# ========== Base Class =========
class Base(DeclarativeBase):
  """
    Base class para todos os modelos ORM.

    Fornece o tipo Mapped para type hints modernos (SQLAlchemy 2.0 style).
  """
  pass

# ========== Enum Classes =========
class ProcessStatus(str, enum.Enum):
  """Estados possíveis de um Process durante o ciclo de vida."""
  CREATED = "created"
  FILES_UPLOADED = "files_uploaded"
  PROCESSING = "processing"
  COMPLETED = "completed"
  FAILED = "failed"
  CANCELLED = "cancelled"

class ParseStatus(str, enum.Enum):
  """Estados da extração de um texto de um CV."""
  PENDING = "pending"
  SUCCESS = "success"
  ERROR = "error"

class UserRole(str, enum.Enum):
  """Funções de utilizador no sistema."""
  ADMIN = "admin"
  RECRUITER = "recruiter"

# ========== Models =========
class Process(Base):
  """
    Representa uma rodada de screening -> um processo de recrutamento.
    
    Um Process contém múltiplos Candidates (CVs) e seus Results (pontuações).
    Ciclo de vida: created → files_uploaded → processing → completed|failed
  """

  __tablename__= "processes"

  id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
  )

  title: Mapped[str] = mapped_column(String(255), nullable=False)

  jd_text: Mapped[str] = mapped_column(Text, nullable=False)

  status: Mapped[ProcessStatus] = mapped_column(
    Enum(ProcessStatus),
    nullable=False,
    default=ProcessStatus.CREATED,
    index=True
  )

  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    nullable=False
  )

  updated_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    onupdate=lambda: datetime.now(timezone.utc),
    nullable=False
  )

  completed_at: Mapped[Optional[datetime]] = mapped_column(
    DateTime(timezone=True),
    nullable=True
  )

  error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

  # Relacionamentos
  candidates: Mapped[List["Candidate"]] = relationship(
    back_populates="process",
    cascade="all, delete-orphan",
    foreign_keys="Candidate.process_id"
  )

  errors: Mapped[List["ProcessingError"]] = relationship(
    back_populates="process",
    cascade="all, delete-orphan",
    foreign_keys="ProcessingError.process_id"
  )

# Coluna Candidate
class Candidate(Base):
  """
    Representa um CV submetido a um Process.
    
    Armazena metadados do ficheiro e o texto extraído (raw_text).
    raw_text é usado internamente pelo pipeline v1.0, não é retornado pela API.
    """

  __tablename__ = "candidates"

  id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
  )

  process_id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("processes.id", ondelete="CASCADE"),
    nullable=False,
    index=True
  )

  name: Mapped[str] = mapped_column(String(255), nullable=False)

  original_filename: Mapped[str] = mapped_column(String(255), nullable=False)

  stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)

  file_path: Mapped[str] = mapped_column(Text, nullable=False)

  raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

  parse_status: Mapped[ParseStatus] = mapped_column(
    Enum(ParseStatus),
    nullable=False,
    default=ParseStatus.PENDING,
    index=True
  )

  parse_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    nullable=False
  )

  # Relacionamentos
  process: Mapped[List["Process"]] = relationship(back_populates="candidates")

  result: Mapped[Optional["Result"]] = relationship(
    back_populates="candidate",
    cascade="all, delete-orphan",
    uselist=False,
    foreign_keys="Result.candidate_id"
  )

  errors: Mapped[List["ProcessingError"]] = relationship(
    back_populates="candidate",
    cascade="all, delete-orphan",
    foreign_keys="ProcessingError.candidate_id"
  )

# Coluna Result
class Result(Base):
  """
    Resultado da pontuação de um Candidate.
    
    Armazena total_score, categoria, breakdown detalhado, e listas de skills.
    breakdown é JSON com scores por critério (skills_match, experience_years, etc).
  """

  __tablename__ = "results"

  id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
  )

  candidate_id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("candidates.id", ondelete="CASCADE"),
    nullable=False,
    index=True
  )

  total_score: Mapped[float] = mapped_column(Float, nullable=False)

  category: Mapped[str] = mapped_column(String(50), nullable=False)

  breakdown: Mapped[dict] = mapped_column(JSON, nullable=False)

  matched_skills: Mapped[List] = mapped_column(JSON, nullable=False)

  required_skills: Mapped[List] = mapped_column(JSON, nullable=False)

  experience_years_found: Mapped[int] = mapped_column(Integer, nullable=False)

  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    nullable=False
  )

  # Relacionamentos
  candidate: Mapped["Candidate"] = relationship(back_populates="result")

class ProcessingError(Base):
  """
    Log de erros que ocorreram durante screening de um Candidate ou Process.
    
    process_id é sempre definido. candidate_id é opcional (erros podem ser a nível de processo).
    stage indica em que etapa do pipeline ocorreu o erro.
  """

  __tablename__ = "processing_errors"

  id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
  )

  process_id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("processes.id", ondelete="CASCADE"),
    nullable=False,
    index=True
  )

  candidate_id: Mapped[Optional[uuid.UUID]] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("candidates.id", ondelete="CASCADE"),
    nullable=True,
    index=True
  )

  stage: Mapped[str] = mapped_column(String(50), nullable=False)

  message: Mapped[str] = mapped_column(Text, nullable=False)

  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    nullable=False
  )

  # Relacionamentos
  process: Mapped["Process"] = relationship(back_populates="errors")
  candidate: Mapped[Optional["Candidate"]] = relationship(back_populates="errors")

class User(Base):
  """
    Utilizador do sistema (autenticação JWT).
    
    Email é unique. Senha é hashed com bcrypt (passlib).
    role pode ser 'recruiter' ou 'admin'. is_active controla acesso.
  """

  __tablename__ = "users"
  
  id: Mapped[uuid.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    default=uuid.uuid4,
  )

  email: Mapped[str] = mapped_column(
    String(255), 
    nullable=False, 
    unique=True,
    index=True
  )

  hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

  role: Mapped[UserRole] = mapped_column(
    Enum(UserRole),
    default= UserRole.RECRUITER,
    nullable=False
  )

  is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    nullable=False
  )