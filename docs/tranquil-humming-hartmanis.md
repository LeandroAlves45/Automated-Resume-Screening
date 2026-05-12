# Plano: Estrutura e Sprints v2.0

## Context

O v1.0 esta completo como CLI pipeline (63 testes passando). O v2.0 e uma API FastAPI que
envolve o v1.0 sem o modificar. O scaffolding de pastas do v2.0 foi criado mas todos os
ficheiros .py ainda estao a 0 bytes. O objetivo e validar a estrutura e definir a ordem
dos sprints de implementacao.

---

## Estado Real do Projeto

### v1.0 (backend/) — Completo e funcional

| Ficheiro | Estado |
|---|---|
| config.py | Implementado — pesos, thresholds, extensoes suportadas |
| main.py | Implementado — CLI entry point, pipeline orquestrador |
| parser/resume_parser.py | Implementado — PDF, DOCX, TXT via Facade pattern |
| parser/jd_parser.py | Implementado — extrai skills, experiencia, educacao, keywords |
| nlp/preprocessor.py | Implementado — limpeza de texto, tokenizacao spaCy |
| nlp/extractor.py | Implementado — features: skills matched, exp years, edu level |
| scoring/scorer.py | Implementado — scoring ponderado TF-IDF + cosine similarity |
| reports/reporter.py | Implementado — CSV, JSON, TXT, terminal |
| tests/ | 63 testes passando (unit + integration) |

Bug corrigido: reporter.py metodo chama-se save_text(), nao save_txt() — verificar em main.py.

### v2.0 (backend/api/) — Scaffolding criado, ficheiros vazios

| Ficheiro | Estado |
|---|---|
| api/config.py | Criado mas VAZIO |
| api/db/database.py | Criado mas VAZIO |
| api/db/models.py | Criado mas VAZIO |
| api/db/migrations/ | Pasta criada, sem ficheiros Alembic |
| api/models/schemas.py | Criado mas VAZIO |
| api/routes/auth.py | Criado mas VAZIO |
| api/routes/processes.py | Criado mas VAZIO |
| api/routes/upload.py | Criado mas VAZIO |
| api/routes/results.py | Criado mas VAZIO |
| api/services/auth_service.py | Criado mas VAZIO |
| api/services/candidate_service.py | Criado mas VAZIO |
| api/services/process_service.py | Criado mas VAZIO |
| api/services/screening_service.py | Criado mas VAZIO |
| api/services/report_service.py | Criado mas VAZIO |
| api/utils/errors.py | Criado mas VAZIO |
| api/utils/logging.py | Criado mas VAZIO |
| api/utils/validators.py | Criado mas VAZIO |

### Infraestrutura

| Ficheiro | Estado |
|---|---|
| requirements.txt (raiz) | Completo — FastAPI, SQLAlchemy, Alembic, JWT, psycopg2 |
| requirements-dev.txt | Completo — pytest-asyncio, httpx, ruff |
| .env.example | Completo — DATABASE_URL, JWT, STORAGE_PATH, CORS |
| .github/workflows/ci.yml | Existe |
| .github/workflows/deploy.yml | Existe |
| docker-compose.yml | NAO EXISTE — gap critico |
| alembic.ini | NAO EXISTE — gap critico |
| frontend/src/ | Pasta existe, sem implementacao |
| backend/requirements.txt | DUPLICADO — apagar ou renomear |

---

## Avaliacao das Camadas

### Fluxo de Dependencias (unidirecional — correto)

```
HTTP Request
    |
    v
routes/         — recebe HTTP, valida via Pydantic, chama servico
    |
    v
services/       — logica de negocio, orquestra DB e pipeline
    |         |
    v         v
db/models   backend/ (v1.0 pipeline — nao modificado)
    |
    v
PostgreSQL
```

### Regras de camada verificadas

- Rotas nao acedem ao DB diretamente (correto)
- Servicos nao conhecem as rotas (correto)
- O pipeline v1.0 e chamado apenas por screening_service (correto)
- Schemas Pydantic sao independentes dos ORM models (correto)

---

## Gaps Criticos (bloqueiam inicio dos sprints)

1. **docker-compose.yml ausente** — sem PostgreSQL local nao e possivel desenvolver
2. **backend/api/main.py ausente** — nao existe o entrypoint FastAPI
3. **alembic.ini ausente** — sem Alembic nao ha migrations

---

## Ordem de Implementacao por Sprint

### Sprint 0 — Infraestrutura (pre-requisito absoluto)

Objetivo: ambiente local a correr com FastAPI + PostgreSQL

Ficheiros a criar:

- docker-compose.yml (servicos: api, postgres)
- backend/api/main.py (app = FastAPI(), include_router, startup events, health check)
- backend/api/config.py (Settings via pydantic-settings, le DATABASE_URL, JWT_SECRET, etc.)
- alembic.ini + backend/api/db/migrations/env.py

Criterio de conclusao: uvicorn sobe, GET /api/health retorna 200, PostgreSQL acessivel.

### Sprint 1 — Camada de Dados

Objetivo: schema da base de dados definido e migravel

Ficheiros a implementar:

- backend/api/db/database.py (engine, SessionLocal, get_db() dependency)
- backend/api/db/models.py (ORM: Process, Candidate, Result, ProcessingError, User)
- Primeira migration Alembic: tabelas base

Criterio de conclusao: alembic upgrade head cria todas as tabelas, alembic downgrade reverte.

### Sprint 2 — Schemas e Utils

Objetivo: contratos de API definidos, tratamento de erros uniforme

Ficheiros a implementar:

- backend/api/models/schemas.py (todos os Pydantic request/response models)
- backend/api/utils/errors.py (NotFoundError, ConflictError, ValidationError, etc.)
- backend/api/utils/logging.py (setup logging, request_id middleware)
- backend/api/utils/validators.py (validacao de ficheiros, MIME types)

Criterio de conclusao: schemas importaveis sem erros, errors retornam JSON consistente.

### Sprint 3 — Servicos Core

Objetivo: logica de processos e upload de CVs funcional

Ficheiros a implementar:

- backend/api/services/process_service.py (CRUD + maquina de estados)
- backend/api/services/candidate_service.py (upload, validacao MIME, registo no DB)
- backend/api/routes/processes.py (POST/GET /api/processes)
- backend/api/routes/upload.py (POST /api/processes/{id}/upload)

Criterio de conclusao: criar processo, fazer upload de CV, ver lista de candidatos.

### Sprint 4 — Autenticacao

Objetivo: todos os endpoints protegidos por JWT

Ficheiros a implementar:

- backend/api/services/auth_service.py (register, login, get_current_user, JWT)
- backend/api/routes/auth.py (POST /api/auth/register, POST /api/auth/login)

Criterio de conclusao: endpoints sem token retornam 401, login retorna JWT valido.

### Sprint 5 — Screening e Results

Objetivo: pipeline v1.0 acessivel via API, resultados persistidos

Ficheiros a implementar:

- backend/api/services/screening_service.py (orquestra process_resumes() + sort_results())
- backend/api/routes/results.py (POST /api/processes/{id}/run, GET /api/processes/{id}/results)

Criterio de conclusao: POST /run dispara pipeline em background, GET /results retorna ranking.

### Sprint 6 — Reports

Objetivo: exportacao de resultados em CSV, JSON, TXT

Ficheiros a implementar:

- backend/api/services/report_service.py (gera e retorna ficheiros via StreamingResponse)

Criterio de conclusao: download de CSV/JSON/TXT com resultados reais.

### Sprint 7 — Testes e Seguranca

Objetivo: cobertura minima 70%, testes de seguranca passando

Ficheiros a criar:

- backend/api/tests/integration/ (endpoints + DB real)
- backend/api/tests/security/ (JWT invalido, path traversal, ficheiros maliciosos)

Criterio de conclusao: pytest com TEST_DATABASE_URL passa, sem falhas de seguranca.
