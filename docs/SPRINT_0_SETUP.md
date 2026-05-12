AUTOMATED RESUME SCREENER - SPRINT 0 SETUP

Objetivo: ter FastAPI + PostgreSQL a correr localmente com Docker Compose, uvicorn, SQLAlchemy e Alembic.

========== FICHEIROS PRINCIPAIS ==========

- backend/docker-compose.yml - Orquestra PostgreSQL 15 + FastAPI.
- backend/Dockerfile - Imagem Docker com Python 3.11, dependencias e modelo spaCy.
- backend/.env.example - Template de variaveis.
- backend/.env - Variaveis locais de development. Nao fazer commit.
- backend/api/config.py - Le variaveis de ambiente via Pydantic.
- backend/api/db/database.py - SQLAlchemy engine, SessionLocal, get_db() e health check DB.
- backend/api/db/models.py - ORM models: Process, Candidate, Result, ProcessingError, User.
- backend/api/main.py - FastAPI app, CORS, lifespan, health check.
- backend/alembic.ini - Configuracao Alembic.
- backend/alembic/env.py - Environment Alembic ligado aos modelos ORM.

========== EXECUTAR LOCALMENTE ==========

Pre-requisitos:

- Docker Desktop a correr.
- Docker Compose disponivel via `docker compose`.

Passos:

1. Navega ate a pasta do backend:

   ```powershell
   cd "C:\Users\Leandro Alves\Desktop\Projetos\Automated Resume Screening\Project\resume-screener\backend"
   ```

2. Sobe os containers:

   ```powershell
   docker compose up --build
   ```

   Isto vai:
   - Construir a imagem da API.
   - Instalar dependencias Python.
   - Fazer download do modelo `en_core_web_sm`.
   - Iniciar PostgreSQL.
   - Iniciar FastAPI com uvicorn.

3. Aguarda ate veres:

   ```text
   Application startup complete.
   Uvicorn running on http://0.0.0.0:8000
   ```

4. Testa o health check noutro terminal:

   ```powershell
   Invoke-WebRequest -UseBasicParsing http://localhost:8000/api/health
   ```

   Ou:

   ```powershell
   curl http://localhost:8000/api/health
   ```

   Resposta esperada:

   ```json
   {
     "status": "ok",
     "version": "2.0.0",
     "database": "connected",
     "nlp_model": "loaded",
     "environment": "development"
   }
   ```

5. Acede a documentacao interativa:

   ```text
   http://localhost:8000/docs
   http://localhost:8000/redoc
   ```

========== ESTRUTURA DO CONTAINER ==========

O container da API usa:

```text
WORKDIR /app
PYTHONPATH=/app
```

O `docker-compose.yml` monta a raiz do projeto em `/app`:

```yaml
volumes:
  - ..:/app
```

Isto e importante porque o codigo importa o package `backend`, por exemplo:

```python
from backend.api.config import get_settings
```

Se fosse montado apenas `.:/app`, o container teria `/app/api` em vez de `/app/backend/api`, causando:

```text
ModuleNotFoundError: No module named 'backend'
```

========== ALTERACOES QUE NAO PRECISAM DE REBUILD ==========

Como o projeto esta montado por volume, alteracoes em ficheiros como estes aparecem logo dentro do container:

- `backend/api/main.py`
- `backend/api/db/models.py`
- `backend/alembic.ini`
- `backend/alembic/env.py`
- ficheiros de codigo Python em geral

Nestes casos, normalmente nao precisas de `docker compose up --build`.

Precisas de rebuild quando alteras:

- `backend/Dockerfile`
- `requirements.txt`
- dependencias Python
- pacotes de sistema instalados via `apt-get`
- passos de build, como download do modelo spaCy

========== CRIAR A PRIMEIRA MIGRATION ==========

Com os containers a correr, entra no container da API:

```powershell
docker compose exec api bash
```

Se `bash` nao estiver disponivel:

```powershell
docker compose exec api sh
```

Dentro do container:

```bash
cd /app/backend
alembic revision --autogenerate -m "Initial schema: Process, Candidate, Result, User"
alembic upgrade head
```

Alternativa se quiseres continuar em `/app`:

```bash
alembic -c backend/alembic.ini revision --autogenerate -m "Initial schema: Process, Candidate, Result, User"
alembic -c backend/alembic.ini upgrade head
```

Isto gera um ficheiro em:

```text
backend/alembic/versions/
```

E aplica as tabelas no PostgreSQL.

Nota: a app ainda chama `Base.metadata.create_all(bind=engine)` no startup. Se a base de dados ja tiver tabelas, o `alembic revision --autogenerate` pode gerar uma migration vazia porque o Alembic compara os modelos com uma base de dados que ja esta atualizada.

========== VERIFICAR TABELAS ==========

Fora do container da API, no PowerShell:

```powershell
docker compose exec postgres psql -U ars_leandro -d ars_dev -c "\dt"
```

Tabelas esperadas:

- `processes`
- `candidates`
- `results`
- `processing_errors`
- `users`
- `alembic_version`, quando migrations forem aplicadas

========== PARAR DOCKER COMPOSE ==========

Para parar os containers mantendo os dados:

```powershell
docker compose down
```

Para parar e remover volumes, apagando dados do banco:

```powershell
docker compose down -v
```

========== TROUBLESHOOTING ==========

Erro:

```text
ModuleNotFoundError: No module named 'backend'
```

Solucao: confirmar que o volume da API monta a raiz do projeto:

```yaml
- ..:/app
```

Erro:

```text
FAILED: No 'script_location' key found in configuration.
```

Solucao: confirmar que `backend/alembic.ini` tem:

```ini
[alembic]
script_location = %(here)s/alembic
```

E correr Alembic a partir de `/app/backend` ou com `-c backend/alembic.ini`.

Erro:

```text
ImportError: cannot import name 'engine' from 'backend.api.db.models'
```

Solucao: `engine` deve ser importado de `backend.api.db.database`, enquanto `Base` vem de `backend.api.db.models`.

Erro:

```text
psycopg2 connection failed
```

Solucao: aguardar o healthcheck do PostgreSQL e ver logs:

```powershell
docker compose logs postgres
```

Erro:

```text
spaCy model not found
```

Solucao: o Dockerfile ja faz download. Se necessario:

```bash
python -m spacy download en_core_web_sm
```

========== PROXIMOS PASSOS ==========

Sprint 0 esta completo quando:

- `docker compose up --build` sobe PostgreSQL e API.
- `GET /health` retorna 200 OK.
- Alembic consegue ler a configuracao.
- Migrations conseguem ser geradas e aplicadas.
- As tabelas existem no PostgreSQL.

Proximo: Sprint 1 - implementar camada de servicos.
