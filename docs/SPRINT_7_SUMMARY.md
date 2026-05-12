# Sprint 7 — Testes e Segurança — OVERVIEW RÁPIDO

**Ficheiro completo:** `SPRINT_7_TEST_PLAN.md` (400+ linhas, plano detalhado)

---

## Estatísticas Rápidas

| Métrica | Valor |
|---|---|
| **Total de testes** | ~80-100 |
| **Unit tests** | ~50 |
| **Integration tests** | ~14 |
| **Security tests** | ~18 |
| **Load tests** | ~4 |
| **Tempo execução** | 5-10 min (CI) |
| **Cobertura alvo** | 70%+ |
| **Ficheiros de teste** | 15 arquivos .py |

---

## Matriz de Testes por Componente

```
SERVICES (5 ficheiros, ~30 testes)
├─ auth_service.py (6 testes)
│  └─ register, login, JWT validation, password hashing
├─ process_service.py (8 testes)
│  └─ CRUD, ownership, state transitions, cascade delete
├─ candidate_service.py (7 testes)
│  └─ file upload, MIME validation, file size, ownership
├─ screening_service.py (5 testes)
│  └─ run screening, results, state transitions
└─ report_service.py (8 testes)
   └─ CSV/JSON/TXT export, tempfile cleanup, format validation

ROUTES (4 ficheiros, ~29 testes)
├─ auth_routes.py (6 testes)
│  └─ POST /register, POST /login, JWT required
├─ processes_routes.py (7 testes)
│  └─ CRUD /api/processes, ownership, 403 forbidden
├─ upload_routes.py (6 testes)
│  └─ POST /upload, MIME/size validation, 413 limits
└─ results_routes.py (10 testes)
   └─ POST /run, GET /results, GET /export/{csv,json,txt}

UTILS (2 ficheiros, ~9 testes)
├─ validators.py (5 testes)
│  └─ UUID, MIME type, file size validation
└─ errors.py (4 testes)
   └─ JSON responses, status codes, error messages

INTEGRATION (4 ficheiros, ~14 testes)
├─ full_flow_screening.py (3 testes)
│  └─ register → create → upload → run → results
├─ full_flow_export.py (3 testes)
│  └─ screening → export CSV/JSON/TXT validation
├─ state_transitions.py (4 testes)
│  └─ CREATED → PROCESSING → COMPLETED, prevent double-run
└─ database_integrity.py (4 testes)
   └─ cascade delete, foreign keys, unique constraints

SECURITY (5 ficheiros, ~18 testes)
├─ jwt_validation.py (5 testes)
│  └─ expired token, invalid signature, missing header
├─ file_upload_security.py (5 testes)
│  └─ block executable, MIME mismatch, file size limit
├─ path_traversal.py (3 testes)
│  └─ ../ in filename, admin path, SQL in process_id
├─ sql_injection.py (3 testes)
│  └─ email injection, jd_text injection, parameterized queries
└─ xss_prevention.py (3 testes)
   └─ headers, JSON escaping, CSV formula injection

LOAD (1 ficheiro, ~4 testes)
└─ performance.py (4 testes)
   └─ 1000 candidates export, 100 concurrent, large files, DB indexes
```

---

## Cobertura Alvo por Sprint

```
Sprint 0 — Infrastructure
├─ docker-compose, main.py, config.py
└─ Test: manual (GET /api/health → 200)

Sprint 1 — Database
├─ Models (Process, Candidate, Result, User)
└─ Test: alembic upgrade/downgrade ✓

Sprint 2 — Schemas & Utils
├─ Pydantic models, error handling, validators
└─ Unit: 9 testes (schemas importáveis, errors JSON) ✓

Sprint 3 — Process & Upload
├─ process_service, candidate_service, routes
└─ Unit: 8+7=15 testes, Integration: 1, Security: 5 ✓

Sprint 4 — Auth
├─ auth_service, auth_routes
└─ Unit: 6+6=12 testes, Security: 5 (JWT validation) ✓

Sprint 5 — Screening
├─ screening_service, results_routes
└─ Unit: 5+10=15 testes, Integration: 3 ✓

Sprint 6 — Reports
├─ report_service, export_routes
└─ Unit: 8 testes, Integration: 3 ✓

Sprint 7 — Tests & Security (THIS)
└─ All: 80+ testes, 70%+ coverage, security pass ✓
```

---

## Quick Start: Ficheiro conftest.py

```python
# backend/api/tests/conftest.py
# Fixtures reutilizáveis:
- test_db() → in-memory SQLite
- client() → FastAPI TestClient
- test_user() → user pre-registado
- test_process() → process com status=CREATED
- test_candidates() → 5 candidates
- jwt_token() → JWT válido
- auth_header() → {"Authorization": "Bearer ..."}
```

---

## Executar Testes

```bash
# Todos os testes (unit + integration + security)
pytest backend/api/tests/ -v --cov=backend/api --cov-fail-under=70

# Apenas unit tests
pytest backend/api/tests/unit/ -v

# Apenas integration tests
pytest backend/api/tests/integration/ -v

# Apenas security tests
pytest backend/api/tests/security/ -v

# Com coverage report HTML
pytest backend/api/tests/ --cov=backend/api --cov-report=html
# Abrir htmlcov/index.html no browser
```

---

## Prioridades de Implementação

### Fase 1 (CRÍTICA) — Services + Routes
- ✅ test_auth_service.py (register, login, JWT)
- ✅ test_auth_routes.py (endpoints protection)
- ✅ test_process_service.py (CRUD + ownership)
- ✅ test_candidate_service.py (file upload)
- ✅ test_screening_service.py (run screening)
- ✅ test_report_service.py (export)
- ✅ test_results_routes.py (all endpoints)

**Resultado esperado:** 60+ testes passing, ~60% coverage

### Fase 2 (IMPORTANTE) — Integration + Security
- ✅ test_full_flow_screening.py (end-to-end)
- ✅ test_jwt_validation.py (token security)
- ✅ test_file_upload_security.py (malicious files)
- ✅ test_sql_injection.py (parameterized queries)

**Resultado esperado:** 80+ testes passing, ~70% coverage

### Fase 3 — Load Tests
- ⚠️ test_performance.py (1000 candidates, concurrent)

**Resultado esperado:** 85+ testes passing, performance baseline

---

## Exemplo: Um Teste Completo

```python
# backend/api/tests/unit/services/test_auth_service.py

import pytest
from backend.api.services.auth_service import AuthService
from backend.api.utils.errors import ValidationError, UnauthorizedError

@pytest.fixture
def auth_service(test_db):
    return AuthService(db=test_db)

def test_register_user_success(auth_service):
    """Registra novo utilizador com hash de password"""
    
    # ARRANGE
    email = "test@example.com"
    password = "SecurePassword123!"
    
    # ACT
    user = auth_service.register(email, password)
    
    # ASSERT
    assert user.email == email
    assert user.password_hash != password  # Never plaintext
    assert user.password_hash.startswith("$argon2")  # Argon2 hash
    
def test_register_user_duplicate_email(auth_service, test_user):
    """Rejeita duplicate email"""
    
    # ARRANGE: test_user já existe
    
    # ACT & ASSERT
    with pytest.raises(ConflictError) as exc_info:
        auth_service.register(test_user.email, "AnotherPassword123!")
    
    assert "already exists" in str(exc_info.value)

def test_login_success(auth_service, test_user):
    """Login com credenciais corretas retorna JWT"""
    
    # ACT
    token = auth_service.login(test_user.email, "correct_password")
    
    # ASSERT
    assert token.startswith("eyJ")  # JWT format
    # Verificar conteúdo do JWT
    decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    assert decoded["user_id"] == test_user.id
```

---

## Checklist Final

### Antes de Deploy:

- [ ] Todos os 80+ testes passando (`pytest ... --cov-fail-under=70`)
- [ ] Coverage report gerado (htmlcov/index.html)
- [ ] Security tests passando (jwt, file upload, SQL injection, XSS)
- [ ] Load tests baseline estabelecido (export 1000 candidates < 5s)
- [ ] CI/CD pipeline configurado (.github/workflows/ci.yml)
- [ ] README.md com instruções de teste
- [ ] Documentação de como rodar testes localmente

### Code Quality:

- [ ] Todos os imports no conftest.py funcionam
- [ ] Fixtures são reutilizáveis e não têm side effects
- [ ] Test names são descritivos (test_<feature>_<scenario>)
- [ ] Arrange-Act-Assert pattern seguido
- [ ] No hardcoded test data (usar fixtures)

---

## Próximos Passos

1. **Criar conftest.py** com todas as fixtures
2. **Implementar testes Fase 1** (services + routes) — ~50 testes
3. **Rodar pytest locally**, iterar até 70% coverage
4. **Implementar testes Fase 2** (integration + security) — +30 testes
5. **Configure CI/CD** to run tests on every push
6. **Deploy com confiança** ✅

---

**Sprint 7 ready to go! 🚀**
