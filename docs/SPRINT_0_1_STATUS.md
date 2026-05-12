SPRINT 0 e 1 - STATUS

Status: COMPLETO E VALIDADO

Data de validacao: April 2026

========== RESUMO ==========

Sprint 0 deixou a infraestrutura base a funcionar:

- FastAPI arranca dentro de Docker.
- PostgreSQL 15 arranca e fica healthy.
- API conecta ao PostgreSQL.
- Modelo spaCy `en_core_web_sm` carrega no startup.
- Endpoint `GET /health` responde `200 OK`.
- Alembic esta configurado para migrations.
- Estrutura de imports `backend.*` funciona dentro do container.

========== VALIDACAO EXECUTADA ==========

Comando:

```powershell
docker compose up -d --build
```

Resultado:

- Imagem `backend-api` construida com sucesso.
- Container `ars_postgres` iniciado e healthy.
- Container `backend-api-1` iniciado.
- Uvicorn a correr em `http://0.0.0.0:8000`.

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health
```

Resposta validada:

```json
{
  "status": "ok",
  "version": "2.0.0",
  "database": "connected",
  "nlp_model": "loaded",
  "environment": "development"
}
```

========== FICHEIROS ATUALIZADOS ==========

Infraestrutura:

- `backend/docker-compose.yml`
- `backend/Dockerfile`
- `.dockerignore`

API:

- `backend/api/main.py`
- `backend/api/config.py`
- `backend/api/db/database.py`
- `backend/api/db/models.py`

Migrations:

- `backend/alembic.ini`
- `backend/alembic/env.py`
- `backend/alembic/script.py.mako`

Documentacao:

- `docs/SPRINT_0_SETUP.md`
- `docs/SPRINT_0_STATUS.md`

========== CORRECOES IMPORTANTES ==========

1. Docker volume da API

Problema anterior:

```yaml
volumes:
  - .:/app
```

Isto montava apenas a pasta `backend` em `/app`, fazendo o container ver:

```text
/app/api
/app/nlp
/app/parser
```

Mas o codigo usa imports como:

```python
from backend.api.config import get_settings
```

Resultado:

```text
ModuleNotFoundError: No module named 'backend'
```

Correcao:

```yaml
volumes:
  - ..:/app
```

Agora o container ve:

```text
/app/backend/api
/app/backend/nlp
/app/backend/parser
```

E `PYTHONPATH=/app` permite importar `backend.*`.

2. Remocao de `version` no Docker Compose

O Compose v2 ignora a chave `version` e mostrava warning:

```text
the attribute `version` is obsolete
```

Foi removida para evitar confusao.

3. Import de `engine` no `main.py`

Problema anterior:

```python
from backend.api.db.models import Base, engine
```

Mas `engine` esta em `backend.api.db.database`.

Correcao:

```python
from backend.api.db.database import check_db_connection, engine
from backend.api.db.models import Base
```

4. Configuracao Alembic

Problema anterior:

```text
FAILED: No 'script_location' key found in configuration.
```

Correcao em `backend/alembic.ini`:

```ini
[alembic]
script_location = %(here)s/alembic
```

O uso de `%(here)s` permite chamar Alembic tanto a partir de `/app/backend` como a partir de `/app` com `-c backend/alembic.ini`.

========== COMO CORRER ALEMBIC ==========

Entrar no container:

```powershell
cd "C:\Users\Leandro Alves\Desktop\Projetos\Automated Resume Screening\Project\resume-screener\backend"
docker compose exec api bash
```

Dentro do container:

```bash
cd /app/backend
alembic revision --autogenerate -m "Initial schema: Process, Candidate, Result, User"
alembic upgrade head
exit
```

Alternativa a partir de `/app`:

```bash
alembic -c backend/alembic.ini revision --autogenerate -m "Initial schema: Process, Candidate, Result, User"
alembic -c backend/alembic.ini upgrade head
```

Verificar tabelas:

```powershell
docker compose exec postgres psql -U ars_leandro -d ars_dev -c "\dt"
```

Tabelas esperadas:

- `processes`
- `candidates`
- `results`
- `processing_errors`
- `users`
- `alembic_version`, quando Alembic aplicar migrations

========== NOTA SOBRE REBUILD ==========

Nao e necessario fazer rebuild para alteracoes em ficheiros montados por volume, por exemplo:

- `backend/api/main.py`
- `backend/alembic.ini`
- `backend/alembic/env.py`
- modelos ORM
- rotas e servicos Python

E necessario fazer rebuild quando muda:

- `backend/Dockerfile`
- `requirements.txt`
- dependencias Python
- pacotes de sistema
- passos de build

========== ESTADO ATUAL ==========

Containers:

- `ars_postgres`: up, healthy, porta `5432`.
- `backend-api-1`: up, porta `8000`.

API:

- `GET /` disponivel.
- `GET /health` disponivel e validado.
- Swagger UI disponivel em `/docs`.
- ReDoc disponivel em `/redoc`.

Database:

- PostgreSQL acessivel internamente em `postgres:5432`.
- PostgreSQL acessivel localmente em `localhost:5432`.
- User: `ars_leandro`.
- Database: `ars_dev`.

Alembic:

- Configuracao corrigida.
- `alembic current` consegue ler o config.
- Proximo passo operacional: criar e aplicar a primeira migration.

========== RISCO / NOTA TECNICA ==========

Atualmente a API chama:

```python
Base.metadata.create_all(bind=engine)
```

no startup.

Isto e util para desenvolvimento inicial, mas pode interferir com a primeira migration porque as tabelas podem ja existir antes do Alembic gerar o diff.

Para uma migration inicial limpa, usar uma base de dados vazia ou remover/migrar esse `create_all` quando Alembic passar a ser a fonte oficial do schema.

========== PROXIMOS PASSOS ==========

Sprint 1 - camada de servicos:

- ProcessService.
- CandidateService.
- Validacao de ficheiros.
- Fluxo de processamento.
- Testes unitarios da camada de servicos.

Sprint 0 fica pronto para transicao quando:

- Migration inicial existe em `backend/alembic/versions/`.
- `alembic upgrade head` aplica sem erro.
- `psql \dt` mostra as tabelas esperadas.
