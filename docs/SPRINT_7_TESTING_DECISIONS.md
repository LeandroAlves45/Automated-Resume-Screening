# Sprint 7 — Testing Strategy Decisions

**Data:** 2026-05-05  
**Status:** ✅ Aprovado  
**Objetivo:** Definir estratégia para conftest.py, assertions, e fixtures

---

## ❓ Questão 1: Melhor Estratégia para Respostas HTTP?

### ✅ **RECOMENDAÇÃO: Assertions Estruturadas com Helpers**

```python
# ✅ PADRÃO RECOMENDADO
def assert_csv_response(response, min_rows=2):
    """Helper reutilizável para validar CSV responses"""
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert "attachment" in response.headers["Content-Disposition"]
    assert b"Rank,Candidate," in response.content
    lines = response.content.decode("utf-8").split("\n")
    assert len(lines) >= min_rows

def test_export_csv(client, test_process_completed):
    response = client.get(f"/api/processes/{test_process_completed.id}/export/csv")
    assert_csv_response(response, min_rows=3)
```

**Nota importante (alinhado com o backend):**
- O fixture `client` é um `fastapi.testclient.TestClient`.
- Mas `response = client.get(...)` devolve um **`httpx.Response`** (com `.status_code`, `.headers`, `.json()`, `.text`, `.content`).
- Portanto, os helpers em `helpers/assertions.py` devem tipar `response` como `httpx.Response` (não `TestClient`).

### Vantagens:
- ✅ DRY (não repetir assertions)
- ✅ Consistência entre testes
- ✅ Manutenção centralizada
- ✅ Assertions claras por tipo de response

### Estrutura por Tipo:

```python
# helpers/assertions.py
def assert_success_response(response, expected_status=200):
    assert response.status_code == expected_status

def assert_csv_response(response, min_rows=2):
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert b"Rank,Candidate," in response.content

def assert_json_response(response, expected_keys=None):
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"
    data = response.json()
    if expected_keys:
        assert all(key in data for key in expected_keys)

def assert_error_response(response, status_code, error_key="detail"):
    assert response.status_code == status_code
    data = response.json()
    assert error_key in data

def assert_unauthorized(response):
    assert response.status_code == 401
    # O backend usa mensagens como "Missing authorization token" e "Invalid or expired token"
    msg = str(response.json().get("detail", "")).lower()
    assert ("missing" in msg) or ("invalid" in msg) or ("not authorized" in msg)

def assert_not_found(response):
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
```

**Uso nos testes:**
```python
def test_export_csv_success(client, test_process_completed):
    response = client.get(f"/api/processes/{test_process_completed.id}/export/csv")
    assert_csv_response(response, min_rows=3)

def test_export_csv_not_found(client):
    response = client.get("/api/processes/00000000-0000-0000-0000-000000000000/export/csv")
    assert_not_found(response)

def test_export_csv_unauthorized(client, test_process_completed):
    response = client.get(f"/api/processes/{test_process_completed.id}/export/csv")
    assert_unauthorized(response)
```

---

## ❓ Questão 2: PostgreSQL ou SQLite :memory:?

### ✅ **RECOMENDAÇÃO: SQLite :memory: (Unit Tests) + PostgreSQL Docker (Integration)**

#### Por Type:

| Teste | DB | Razão |
|-------|----|----|
| **Unit Tests** | SQLite :memory: | ⚡ Instantâneo, isolado, perfeito |
| **Integration** | PostgreSQL Docker | 🔧 Realista, full features |
| **Load Tests** | PostgreSQL | 📊 Performance produção |

#### SQLite :memory: (DEFAULT)

```python
@pytest.fixture(scope="function")
def test_db():
    from sqlalchemy import create_engine, event
    
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    
    # ⚠️ CRÍTICO: Ativar foreign keys em SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()
    Base.metadata.drop_all(engine)
```

**Vantagens:**
- ⚡ Testes em < 2 minutos (120 testes)
- ✅ Isolamento perfeito (cada teste tem BD novo)
- 🚀 Zero setup/teardown (em memória)
- 📦 Sem dependências externas

#### PostgreSQL (CI/CD ou Integration)

```bash
# docker-compose.test.yml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: test_db
      POSTGRES_PASSWORD: test_pass
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

# Rodar integration tests
docker-compose -f docker-compose.test.yml up -d
TEST_DB=postgres pytest backend/api/tests/integration/ -v
```

#### Decisão Final:

```python
# conftest.py - HÍBRIDA (flexível)
import os
from sqlalchemy import create_engine

@pytest.fixture(scope="function")
def test_db():
    db_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite:///:memory:"  # ← DEFAULT (rápido)
    )
    
    engine = create_engine(db_url, echo=False)
    
    if "sqlite" in db_url:
        from sqlalchemy import event
        @event.listens_for(engine, "connect")
        def set_pragma(dbapi, _):
            cursor = dbapi.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()
```

**Usar:**
```bash
# ✅ Unit tests (SQLite - rápido)
pytest backend/api/tests/unit/ -v

# ✅ Integration (PostgreSQL - realista)
TEST_DATABASE_URL="postgresql://user:pass@localhost/test_db" \
  pytest backend/api/tests/integration/ -v

# ✅ CI/CD pipeline (SQLite por default)
pytest backend/api/tests/ -v --cov=backend/api
```

---

## ❓ Questão 3: Fixtures Reutilizáveis ou Por-Teste?

### ✅ **RECOMENDAÇÃO: Por-Teste (scope="function") para Isolamento**

#### Matriz de Scopes:

| Fixture | Scope | Razão |
|---------|-------|-------|
| `test_db` | `function` | ✅ Novo DB por teste (isolamento crítico) |
| `client` | `function` | ✅ Novo TestClient por teste |
| `auth_service` | `function` | ✅ Novo service por teste |
| `process_service` | `function` | ✅ Novo service por teste |
| `test_user` | `function` | ✅ Novo user por teste |
| `test_process` | `function` | ✅ Novo process por teste |
| `settings` | `session` | ⚠️ Config imutável, reutilizar |
| `nlp_model` | `session` | ⚠️ Caro carregar, reutilizar (spaCy) |

#### Por quê `scope="function"`?

```python
# ❌ ERRADO - scope="session" (contaminação)
@pytest.fixture(scope="session")
def test_db():
    session = create_session()
    yield session
    # Dados do teste 1 ainda existem para teste 2!

# ✅ CORRETO - scope="function" (isolamento)
@pytest.fixture(scope="function")
def test_db():
    session = create_session()
    yield session
    session.rollback()  # Limpar dados
    # Próximo teste começa clean
```

#### Conftest.py Otimizado:

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from backend.api.main import app
from backend.api.db.models import Base

# ✅ Imutável → session scope (reutilizar)
@pytest.fixture(scope="session")
def settings():
    from backend.api.config import get_settings
    return get_settings()

# ✅ Caro → session scope (reutilizar)
@pytest.fixture(scope="session")
def nlp_model():
    import spacy
    return spacy.load("en_core_web_sm")

# ✅ Crítico isolamento → function scope (novo per teste)
@pytest.fixture(scope="function")
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(engine)

# ✅ Depende de test_db → function scope
@pytest.fixture(scope="function")
def client(test_db):
    def override_get_db():
        yield test_db
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

# ✅ Services → function scope
@pytest.fixture(scope="function")
def auth_service(test_db, settings):
    from backend.api.services.auth_service import AuthService
    return AuthService(db=test_db, settings=settings)

@pytest.fixture(scope="function")
def process_service(test_db):
    from backend.api.services.process_service import ProcessService
    return ProcessService(db=test_db)

# ✅ Test data → function scope
@pytest.fixture(scope="function")
def test_user(client):
    response = client.post("/api/auth/register", json={
        "email": "test@example.com",
        "password": "TestPassword123!"
    })
    return response.json()

@pytest.fixture(scope="function")
def test_process(client, auth_header):
    response = client.post(
        "/api/processes",
        json={"title": "Backend Dev", "jd_text": "5+ years Python"},
        headers=auth_header
    )
    return response.json()

@pytest.fixture(scope="function")
def test_process_completed(test_db, test_process):
    from backend.api.db.models import Process, ProcessStatus
    process = test_db.query(Process).get(test_process["id"])
    process.status = ProcessStatus.COMPLETED
    test_db.commit()
    return process
```

#### Resultado:

```
✅ test_user_1.py → Começa com BD limpo
   - Cria user, process
   - Testa lógica
   - Finaliza (BD descartado)

✅ test_user_2.py → Começa com BD limpo (não vê dados de test_user_1)
   - Cria user (novo ID, mesmos dados)
   - Testa lógica diferente
   - Finaliza (BD descartado)

⚡ Tempo total: < 2 minutos (120 testes)
🔒 Isolamento: 100% (zero interferência entre testes)
```

---

## 📋 Resumo das 3 Decisões

| # | Questão | Decisão | Implementação |
|---|---------|---------|----------------|
| **1** | HTTP Responses | Helpers reutilizáveis | `helpers/assertions.py` |
| **2** | DB Type | SQLite :memory: (default) | `conftest.py` com `PRAGMA foreign_keys=ON` |
| **3** | Fixtures Scope | `scope="function"` (novo per teste) | Todas as fixtures de estado com `function` |

---

## 🚀 Checklist de Implementação

- [ ] Criar `backend/api/tests/helpers/assertions.py` com 6+ helpers
- [ ] Criar `conftest.py` com fixtures `scope="function"`
- [ ] Ativar `PRAGMA foreign_keys=ON` em SQLite
- [ ] Implementar 35 unit tests (services) com assertions.py
- [ ] Implementar 14 route tests com assertions.py
- [ ] Executar: `pytest backend/api/tests/ -v`
- [ ] Validar isolamento: 2 testes devem ser independentes
- [ ] Medir tempo: deve ser < 2 min para 80 testes

---

**Sprint 7 — Testing Strategy FINALIZADA ✅**
