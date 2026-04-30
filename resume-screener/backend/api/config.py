from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
  """
  Configuração centralizada para a aplicação.

  Lê as variáveis de ambiente e fornece valores padrão para parâmetros de configuração.
  """

  # ========= Database =========
  database_url: str = "postgresql://ars_leandro:ars_leandropass@postgres:5432/ars_dev"
  test_database_url: str = "postgresql://ars_user:ars_password@localhost:5432/ars_test"

  # ========= Application =========
  app_env: str = "development"  # development | production | testing
  log_level: str = "DEBUG"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
  debug: bool = True

  # ========= JWT Authentication =========
  jwt_secret_key: str = "change-this-to-a-random-secret-key-in-production"
  jwt_algorithm: str = "HS256"
  jwt_access_token_expire_minutes: int = 30

  # ========= File Upload =========
  storage_path: str = "./storage"
  max_file_size_mb: int = 10
  file_retention_days: int = 30

  # ========= CORS =========
  allowed_origins: str = "http://localhost:3000,http://localhost:5173"

  # ========= NLP =========
  spacy_model: str = "en_core_web_sm"

  class Config:
    """Configurações adicionais para o Pydantic."""
    env_file = ".env"
    env_file_encoding = "utf-8"
    case_sensitive = False

  @property
  def allowed_origins_list(self) -> list[str]:
    """
        Converte string comma-separated em lista de origins válidos para CORS.
        
        Exemplo: "http://localhost:3000,http://localhost:5173" → 
                 ["http://localhost:3000", "http://localhost:5173"]
        """
    return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
  
  @property
  def is_production(self) -> bool:
    """
    Indica se a aplicação está rodando em ambiente de produção.
    Retorna True se app_env for 'production' (case-insensitive), False caso contrário.
    """
    return self.app_env.lower() == "production"
  
  def get_max_file_size_bytes(self) -> int:
    """
    Retorna o tamanho máximo de arquivo permitido para upload, em bytes.
    Converte o valor configurado em megabytes (MB) para bytes (B).
    """
    return self.max_file_size_mb * 1024 * 1024
  
@lru_cache()
def get_settings() -> Settings:
  """
    Dependency Injection Helper para FastAPI.
    
    Carrega Settings uma única vez (em cache) e reutiliza.
    Isto evita ler o .env múltiplas vezes desnecessariamente.
    
    Uso em rotas:
        from backend.api.config import get_settings
        from fastapi import Depends
        
        @router.get("/some-endpoint")
        async def some_endpoint(settings: Settings = Depends(get_settings)):
            ...
    """
  return Settings()