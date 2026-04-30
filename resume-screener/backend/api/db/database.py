from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from backend.api.config import get_settings

# Carrega as configurações do banco de dados
settings = get_settings()

# ========== SQLAlchemy Engine =========
# create_engine com connection pool e timeout configurados
engine = create_engine(
  settings.database_url,
  poolclass=QueuePool,
  pool_size=10,           # Número máximo de conexões no pool
  max_overflow=20,        # Número máximo de conexões extras além do pool_size
  pool_timeout=30,        # Timeout para obter conexão pool (em segundos)
  echo=settings.debug,    # Habilita o log das queries SQL
  connect_args={
    "connect_timeout": 10,  # Tempo máximo para estabelecer uma conexão (em segundos)
    "keepalives": 1,        # Habilita keep-alive para conexões
    "keepalives_idle": 30,  # Tempo de inatividade antes de enviar
  }
)

# ========== Session Factory =========
# SessionLocal é a factory que cria novas sessões de banco de dados
SessionLocal = sessionmaker(
  autocommit=False,   # Não faz commit automático, o commit deve ser feito manualmente
  autoflush=False,    # Não faz flush automático, o flush deve ser feito manualmente
  bind=engine
)

def get_db() -> Generator[Session, None, None]:
  """
    Dependency Injection para FastAPI.
    
    Fornece uma nova sessão de base de dados para cada request.
    A sessão é fechada automaticamente após o request terminar.
    
    Uso em rotas:
        from backend.api.db.database import get_db
        from fastapi import Depends
        from sqlalchemy.orm import Session
        
        @router.get("/example")
        async def example(db: Session = Depends(get_db)):
            result = db.query(SomeModel).all()
            return result
    """
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()

def create_tables() -> None:
  """
    Helper para criar todas as tabelas no banco de dados.
    
    Importa e chama Base.metadata.create_all().
    Usado em testes e inicial setup (alternativa a Alembic para estruturas simples).
    
    Nota: Em produção, usa-se Alembic (alembic upgrade head).
    """
  from backend.api.db.models import Base  # Importa os modelos para registrar as tabelas
  Base.metadata.create_all(bind=engine)

def check_db_connection() -> dict[str, bool]:
  """
    Verifica se a conexão com PostgreSQL está ativa.
    
    Retorna: {"connected": True/False}
    
    Usado por GET /api/health para confirmar que banco está acessível.
    """
  
  try:
    
    with engine.connect() as connection:
      connection.execute(text("SELECT 1"))
    return {"connected": True}
  
  except Exception:
    return {"connected": False}