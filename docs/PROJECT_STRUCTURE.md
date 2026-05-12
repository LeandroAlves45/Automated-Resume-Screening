# Project Structure — Automated Resume Screener

**Version**: v2.0 Phase 2
**Last Updated**: May 2026
**Status**: API backend ~85% complete, frontend pending

---

## Directory Tree

```
resume-screener/                        # Project root
|
|-- requirements.txt                    # v2.0 full stack (FastAPI, SQLAlchemy, JWT, spaCy...)
|-- requirements-dev.txt                # Dev/test tools (pytest-asyncio, httpx, ruff)
|-- .env.example                        # Template for environment variables (commit this)
|-- .env                                # Local secrets — never committed
|-- .gitignore
|
|-- .github/
|   `-- workflows/
|       |-- ci.yml                      # Lint + test on every push/PR
|       `-- deploy.yml                  # Deploy: Render (hook + health) + Vercel (frontend) on push to main
|
|-- frontend/                           # React + TypeScript (v2.0 — not yet implemented)
|   `-- src/
|       |-- App.tsx
|       |-- main.tsx
|       |-- components/                 # Header, FileUploader, ResultsTable, ScoreBreakdown
|       |-- pages/                      # ProcessPage, UploadPage, ResultsPage
|       |-- services/
|       |   `-- api.ts                  # HTTP client wrapper (calls FastAPI)
|       `-- types/
|           `-- index.ts                # TypeScript interfaces matching Pydantic schemas
|
`-- backend/                            # Python backend (v1.0 + v2.0)
    |
    |-- main.py                         # [v1.0] CLI entry point — runs pipeline standalone
    |-- config.py                       # [v1.0] Pipeline constants: weights, thresholds, edu levels
    |-- job_description.txt             # [v1.0] Sample JD for local CLI runs
    |
    |-- parser/                         # [v1.0] File parsing layer
    |   |-- __init__.py
    |   |-- resume_parser.py            # Facade: extracts text from PDF, DOCX, TXT
    |   `-- jd_parser.py                # Extracts skills, experience, education from JD text
    |
    |-- nlp/                            # [v1.0] NLP processing layer
    |   |-- __init__.py
    |   |-- preprocessor.py             # Cleans text, tokenises with spaCy
    |   `-- extractor.py                # Extracts features: matched skills, years, edu level
    |
    |-- scoring/                        # [v1.0] Scoring layer
    |   |-- __init__.py
    |   `-- scorer.py                   # Weighted score: 40% skills, 25% exp, 20% TF-IDF, 15% edu
    |
    |-- reports/                        # [v1.0] Output layer
    |   |-- __init__.py
    |   `-- reporter.py                 # Generates CSV, JSON, TXT reports and terminal output
    |
    |-- tests/                          # [v1.0] Test suite — 63 tests passing
    |   |-- conftest.py                 # Shared fixtures (nlp_model, sample CVs, JD criteria)
    |   |-- unit/
    |   |   |-- test_resume_parser.py
    |   |   |-- test_jd_parser.py
    |   |   |-- test_extractor.py
    |   |   `-- test_scorer.py
    |   `-- integration/
    |       `-- test_pipeline.py        # Full pipeline: parse -> extract -> score -> sort
    |
    `-- api/                            # [v2.0] FastAPI wrapper
        |
        |-- main.py                     # ✅ IMPLEMENTED: FastAPI app, lifespan, routers, middleware
        |-- config.py                   # ✅ IMPLEMENTED: pydantic-settings, environment variables
        |
        |-- db/                         # Data layer
        |   |-- database.py             # ✅ IMPLEMENTED: SQLAlchemy engine, SessionLocal, health checks
        |   |-- models.py               # ✅ IMPLEMENTED: ORM models (User, Process, Candidate, Result)
        |   `-- migrations/             # ✅ IMPLEMENTED: Alembic versions for schema management
        |
        |-- models/                     # Contract layer
        |   `-- schemas.py              # ✅ IMPLEMENTED: Pydantic request/response models (separate from ORM)
        |
        |-- routes/                     # HTTP layer — no business logic
        |   |-- __init__.py
        |   |-- auth.py                 # ✅ IMPLEMENTED: POST /auth/register, POST /auth/login
        |   |-- processes.py            # ✅ IMPLEMENTED: CRUD endpoints for processes
        |   |-- upload.py               # ✅ IMPLEMENTED: POST /processes/{id}/upload with validation
        |   |-- results.py              # ✅ IMPLEMENTED: GET /processes/{id}/results, POST /processes/{id}/run
        |   `-- deps.py                 # ✅ IMPLEMENTED: Dependency injection factories (FastAPI Depends)
        |
        |-- services/                   # Business logic layer
        |   |-- auth_service.py         # ✅ IMPLEMENTED: JWT, password hashing, user authentication
        |   |-- process_service.py      # ✅ IMPLEMENTED: Process CRUD + state machine + reconciliation
        |   |-- candidate_service.py    # ✅ IMPLEMENTED: File upload, MIME validation, filesystem storage
        |   |-- screening_service.py    # ✅ IMPLEMENTED: Calls v1.0 pipeline, persists results
        |   |-- report_service.py       # ✅ IMPLEMENTED: CSV/JSON/TXT export via StreamingResponse
        |   |-- rate_limiter.py         # ✅ IMPLEMENTED: Redis-based rate limiting with fail-open/closed
        |   `-- results_query.py        # ✅ IMPLEMENTED: Query optimization (N+1 fix, result fetching)
        |
        `-- utils/                      # Cross-cutting concerns
            |-- errors.py               # ✅ IMPLEMENTED: Domain exceptions (ValidationError, NotFoundError, etc.)
            |-- logging.py              # ✅ IMPLEMENTED: Logging setup + formatters
            |-- validators.py           # ✅ IMPLEMENTED: File extension, MIME type (python-magic), size validation
            `-- __init__.py
```

---

## Layer Responsibilities

### [v1.0] Pipeline Layers (backend/ — do not modify)

| Layer | Location | Role |
|---|---|---|
| CLI Orchestrator | main.py | Loads spaCy once, runs full pipeline, generates reports |
| Configuration | config.py | Single source of truth for all pipeline constants |
| Parser | parser/ | Extracts plain text from PDF, DOCX, TXT files |
| NLP | nlp/ | Cleans text, tokenises, extracts structured features |
| Scoring | scoring/ | Computes weighted score [0-100] and category |
| Reports | reports/ | Formats and writes output files |

### [v2.0] API Layers (backend/api/ — implemented)

| Layer | Location | Role |
|---|---|---|
| Entry Point | api/main.py | FastAPI app, lifespan, routers, middleware, security headers, health check |
| Config | api/config.py | Environment variables via pydantic-settings |
| ORM | api/db/ | SQLAlchemy models, engine, session, migrations, database health checks |
| Schemas | api/models/ | Pydantic request/response contracts (separate from ORM) |
| Routes | api/routes/ | HTTP handling only — delegates to services |
| Dependencies | api/routes/deps.py | FastAPI dependency injection factories (get_db, get_current_user, etc.) |
| Services | api/services/ | Business logic, DB operations, v1.0 pipeline integration |
| Rate Limiter | api/services/rate_limiter.py | Redis-based rate limiting with configurable fail strategies |
| Query Optimization | api/services/results_query.py | Optimized query patterns (N+1 fix) |
| Utils | api/utils/ | Errors, logging, validators (file validation with python-magic) |

---

## Data Flow

```
Recruiter (browser)
        |
        | HTTP + JWT
        v
  api/routes/          <- validates input via schemas.py, calls service
        |
        v
  api/services/        <- business logic, state transitions, file handling
        |         |
        v         v
  api/db/       backend/ (v1.0 pipeline)
  models.py     parser/ -> nlp/ -> scoring/ -> reports/
        |
        v
  PostgreSQL
```

### Key rule: screening_service imports from v1.0 directly

```python
# backend/api/services/screening_service.py
from backend.parser.resume_parser import ResumeParser
from backend.parser.jd_parser import JobDescriptionParser
from backend.nlp.preprocessor import TextPreprocessor
from backend.nlp.extractor import ResumeFeatureExtractor
from backend.scoring.scorer import ResumeScorer
from backend.main import process_resumes, sort_results
```

The v1.0 pipeline is never modified. It is treated as an internal library.

---

## Process State Machine

```
created
   |
   | (at least one CV uploaded)
   v
files_uploaded
   |
   | POST /processes/{id}/run
   v
processing -----> failed
   |
   | (all candidates processed)
   v
completed
```

---

## Environment Variables (.env.example)

### Core Configuration
| Variable | Used By | Description |
|---|---|---|
| APP_ENV | api/config.py | development / production / testing |
| LOG_LEVEL | api/utils/logging.py | INFO / DEBUG |

### Database
| Variable | Used By | Description |
|---|---|---|
| DATABASE_URL | api/db/database.py | PostgreSQL connection string |
| TEST_DATABASE_URL | pytest | Separate test database for integration tests |

### JWT & Authentication
| Variable | Used By | Description |
|---|---|---|
| JWT_SECRET_KEY | api/services/auth_service.py | Secrets used to sign JWT tokens |
| JWT_ALGORITHM | auth_service.py | Algorithm: HS256 (default) |
| JWT_ACCESS_TOKEN_EXPIRE_MINUTES | auth_service.py | Access token TTL (default: 30) |
| JWT_REFRESH_TOKEN_EXPIRE_DAYS | auth_service.py | Refresh token TTL (default: 7) |

### File Storage & Upload
| Variable | Used By | Description |
|---|---|---|
| STORAGE_PATH | api/services/candidate_service.py | Filesystem path for CV file storage |
| MAX_FILE_SIZE_MB | candidate_service.py | Max upload size in MB (default: 10) |
| FILE_RETENTION_DAYS | candidate_service.py | CV retention policy in days |

### Redis & Rate Limiting
| Variable | Used By | Description |
|---|---|---|
| REDIS_URL | api/services/rate_limiter.py | Redis connection string (redis://localhost:6379) |
| RATE_LIMIT_ENABLED | rate_limiter.py | Enable/disable rate limiting (true/false) |
| RATE_LIMIT_FAIL_OPEN | rate_limiter.py | Fail strategy: true=allow, false=reject (default: true) |
| RATE_LIMIT_LOGIN_REQUESTS | rate_limiter.py | Login attempts limit per window (default: 5) |
| RATE_LIMIT_LOGIN_WINDOW_SECONDS | rate_limiter.py | Login window in seconds (default: 900 = 15 min) |
| RATE_LIMIT_REGISTER_REQUESTS | rate_limiter.py | Register attempts limit per window (default: 3) |
| RATE_LIMIT_REGISTER_WINDOW_SECONDS | rate_limiter.py | Register window in seconds (default: 900 = 15 min) |
| RATE_LIMIT_UPLOAD_REQUESTS | rate_limiter.py | Upload attempts limit per window (default: 10) |
| RATE_LIMIT_UPLOAD_WINDOW_SECONDS | rate_limiter.py | Upload window in seconds (default: 300 = 5 min) |

### CORS & Security
| Variable | Used By | Description |
|---|---|---|
| ALLOWED_ORIGINS | api/main.py | CORS allowed origins (comma-separated list) |

### NLP & Processing
| Variable | Used By | Description |
|---|---|---|
| SPACY_MODEL | api/config.py | spaCy model to load (default: en_core_web_sm) |
| STUCK_PROCESS_TIMEOUT_MINUTES | api/main.py | Timeout for process reconciliation (default: 30) |

---

## Suite de testes (API v2.0)

Inventário actual (**80** testes), árvore de ficheiros, comandos de cobertura, lista do que já está coberto e **roadmap** (upload HTTP, lista de processos, refresh, export, unitários para `validators`, etc.) com o **fluxo da aplicação** passo a passo para implementar cada tipo de teste em falta: ver **[TESTING.md](TESTING.md)**.

- **Unitários:** `backend/api/tests/unit/services/` (auth, process, candidate, report, screening).
- **Integração HTTP:** `backend/api/tests/integration/routes/` (auth, processes, run/results); `integration/routes/conftest.py` injeta o modelo spaCy nas dependências (`get_nlp_model`), porque o `TestClient` não executa o lifespan da app.

---

## Current Status (Phase 2)

### ✅ Completed (Phase 2.1)
- FastAPI REST API (main.py, routers, middleware, security headers)
- Database layer (SQLAlchemy ORM, migrations via Alembic)
- Authentication (JWT with access/refresh tokens, argon2 hashing)
- File upload (MIME validation via python-magic, filesystem storage)
- Rate limiting (Redis-based with configurable rules)
- Unit tests (~80 tests covering services and utilities)
- CI/CD pipelines (lint, test on push)

### 🔄 In Progress (Phase 2.2)
- Integration tests for HTTP routes
- Security tests (CORS, headers, authorization)
- Backend deployment configuration (Docker, Render)

### ⏳ Pending (Phase 3)
- React frontend (TypeScript, components, routing)
- Production deployment (Nginx, load balancing, monitoring)

---

## Notes

- `*.md` files are in `.gitignore` — this file is not committed to the repository.
- `backend/requirements.txt` is a leftover from v1.0 local setup. The canonical file is `requirements.txt` at the project root.
- `storage/` directory (CV file storage) must be created before running the API and added to `.gitignore`.
