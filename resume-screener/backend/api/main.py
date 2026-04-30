import logging
from contextlib import asynccontextmanager
import spacy

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.api.config import get_settings
from backend.api.db.database import check_db_connection, engine
from backend.api.db.models import Base
from backend.api.utils.errors import BaseAPIException

# ========== Logging Setup =========
logger = logging.getLogger(__name__)
logging.basicConfig(
  level=logging.DEBUG,
  format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

# ========== Settings =========
settings = get_settings()

# ========== Startup/Shutdown Events =========
@asynccontextmanager
async def lifespan(app: FastAPI):
  """
  Gerencia o ciclo de vida da aplicação FastAPI.

  Startup:
  - Cria tabelas no banco de dados
  - Carrega modelo spacy 

  Shutdown:
  - Fecha conexões de banco de dados
  """
  logger.info("Starting up Automated Resume Screener...")

  # Cria todas as tabelas do banco de dados
  logger.info("Creating database tables...")
  try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")

  except Exception as e:
    logger.error(f"Error creating database tables: {e}")
    raise

  # Carrega o modelo spacy uma única vez e armazena em app.state
  logger.info(f"Loading spaCy model: {settings.spacy_model}...")
  try:
    nlp = spacy.load(settings.spacy_model)
    app.state.nlp_model = nlp
    logger.info("spaCy model loaded successfully.")

  except OSError as e:
    logger.error(
      f"spaCy model '{settings.spacy_model}' not found. "
      f"Please install it using: python -m spacy download {settings.spacy_model}"
    )
    raise

  yield  # Aplicação roda aqui

  # Shutdown logic 
  logger.info("Shutting down Automated Resume Screener...")

# ========== FastAPI App Initialization ==========
app = FastAPI(
  title="Automated Resume Screener",
  description="REST API for CV screening and ranking",
  version="2.0.0",
  lifespan=lifespan
)

# ========== CORS Middleware ==========
# Configura CORS para permitir requisições do frontend
allowed_origins = settings.allowed_origins_list
logger.info(f"Configuring CORS with allowed origins: {allowed_origins}")

app.add_middleware(
  CORSMiddleware,
  allow_origins=allowed_origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# ========== Response Models ==========
class HealthCheckResponse(BaseModel):
  """ Resposta do health check endpoint. """
  status: str
  version: str
  database: str
  nlp_model: str
  environment: str

# ========== Health Check Endpoint ==========
@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
  """
  Verifica a saúde da aplicação.

  Confirma:
  - Aplicação rodando
  - Conexão com o banco de dados
  - Modelo spaCy carregado

  Response 200: tudo ok
  Response 500: algum componente crítico falhou
  """

  try:
    # Verifica conexão com base de dados
    db_status = check_db_connection()
    database_status = "connected" if db_status["connected"] else "disconnected"

    # Verifica se modelo spaCy está carregado
    nlp_loaded = hasattr(app.state, "nlp_model") and app.state.nlp_model is not None
    nlp_status = "loaded" if nlp_loaded else "not loaded"

    # Se banco ou spaCy falhou, retorna 500
    if not db_status["connected"] or not nlp_loaded:
      raise HTTPException(status_code=500, detail="Health check failed")
    
    return HealthCheckResponse(
      status="ok",
      version="2.0.0",
      database=database_status,
      nlp_model=nlp_status,
      environment=settings.app_env
    )
  except Exception as e:
    logger.error(f"Health check error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
  
# ========== Root Endpoint ==========
@app.get("/api/")
async def root():
  """
  Endpoint raiz -> confirma que a API está rodando.
  """
  return {
    "message": "Automated Resume Screener API is running!",
    "docs": "/docs",
    "openapi": "/openapi.json"
}

# ========== Router Registration ==========
# Importa e regista routers de endpoints
# @app.include_router(auth_router, prefix="/api")
# @app.include_router(processes_router, prefix="/api")
# @app.include_router(upload_router, prefix="/api")
# @app.include_router(results_router, prefix="/api")


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
