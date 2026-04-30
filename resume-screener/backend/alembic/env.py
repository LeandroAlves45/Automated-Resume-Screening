from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, text
from alembic import context
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path para importar backend.
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.config import get_settings
from backend.api.db.models import Base

# Objeto de configuração do Alembic.
config = context.config

# Aplica a configuração de logging definida no ficheiro Alembic.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata usada pela autogeração de migrações.
target_metadata = Base.metadata

# Lê a DATABASE_URL a partir da configuração da aplicação.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

def run_migrations_offline() -> None:
    """
    Corre as migrações no modo 'offline'.

    Configura o contexto diretamente com o URL, sem criar um Engine
    nem exigir um DBAPI específico.

    context.execute() gera SQL diretamente.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """
    Corre as migrações no modo 'online'.

    Cria um Engine e conecta-se à base de dados. O contexto é configurado
    com esta conexão, permitindo que as migrações sejam executadas.
    """

    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = settings.database_url

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()