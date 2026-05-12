# Sprint 7 — Métodos para Testar (ASYNC vs SYNC)

**Objetivo:** Documentar todos os métodos a testar, com indicação de tipo (async/sync) e ajustes necessários.

**Cobertura:** ~120 testes (80 existentes + 40 novos)

---

## 📊 Resumo Executivo

| Camada | ASYNC | SYNC | Total | Ajustes |
|--------|-------|------|-------|---------|
| **Routes** | 12 | 0 | 12 | ✅ TestClient |
| **Services** | 1 | 34 | 35 | ❌ Nenhum |
| **Dependencies** | 2 | 8 | 10 | ✅ TestClient |
| **Utils/Validators** | 0 | 18 | 18 | ❌ Nenhum |
| **Models/ORM** | 0 | 12 | 12 | ❌ Nenhum |
| **TOTAL** | **15** | **72** | **87** | **Estratégia Unificada** |

> **Estratégia:** Usar `TestClient` para todos os testes. Executa async routes de forma síncrona. Zero `@pytest.mark.asyncio`.

---

## 🔴 ROUTES — ASYNC (12 métodos)

> ⚠️ Todos são `async def`. Usar `TestClient(app)` nos testes.

### 1. auth.py (4 rotas ASYNC)

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 1 | `register(register_data, auth_service)` | ASYNC | `test_register_success`, `test_register_duplicate_email`, `test_register_invalid_email`, `test_register_weak_password` | TestClient |
| 2 | `login(credentials, auth_service)` | ASYNC | `test_login_success`, `test_login_wrong_password`, `test_login_user_not_found`, `test_login_missing_fields` | TestClient |
| 3 | `refresh(refresh_req, auth_service)` | ASYNC | `test_refresh_token_success`, `test_refresh_token_expired`, `test_refresh_invalid_token` | TestClient |
| 4 | `logout(logout_req, current_user, auth_service)` | ASYNC | `test_logout_success`, `test_logout_token_blacklist`, `test_logout_already_blacklisted` | TestClient |

### 2. processes.py (3 rotas ASYNC)

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 5 | `create_process(request, process_service, current_user)` | ASYNC | `test_create_process_success`, `test_create_empty_title`, `test_create_empty_jd`, `test_create_unauthorized` | TestClient |
| 6 | `list_processes(process_service, current_user)` | ASYNC | `test_list_processes_empty`, `test_list_processes_paginated`, `test_list_processes_unauthorized`, `test_list_processes_offset_limit` | TestClient |
| 7 | `get_process(process_id, process_service, current_user)` | ASYNC | `test_get_process_success`, `test_get_process_not_found`, `test_get_process_invalid_uuid` | TestClient |

**NOVO (3 rotas adicionais):**

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 8 | `delete_process(process_id, process_service, current_user)` | ASYNC | `test_delete_process_success`, `test_delete_process_not_found`, `test_delete_cascade_candidates` | TestClient |
| 9 | `update_process(process_id, request, process_service, current_user)` | ASYNC | `test_update_process_title`, `test_update_process_jd`, `test_update_nonexistent` | TestClient |
| 10 | `get_process_status(process_id, process_service, current_user)` | ASYNC | `test_get_status_created`, `test_get_status_processing`, `test_get_status_completed` | TestClient |

### 3. results.py (3 rotas ASYNC)

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 11 | `run_screening(process_id, background_tasks, screening_svc)` | ASYNC | `test_run_screening_success`, `test_run_invalid_state`, `test_run_no_candidates` | TestClient |
| 12 | `get_results(process_id, screening_svc)` | ASYNC | `test_get_results_processing`, `test_get_results_completed`, `test_get_results_failed` | TestClient |

**NOVOS (3 rotas adicionais):**

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 13 | `export_csv(process_id, report_svc)` | ASYNC | `test_export_csv_success`, `test_export_csv_not_completed`, `test_export_csv_headers` | TestClient |
| 14 | `export_json(process_id, report_svc)` | ASYNC | `test_export_json_success`, `test_export_json_metadata` | TestClient |

### 4. upload.py (2 rotas ASYNC)

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 15 | `upload_candidates(process_id, file, candidate_svc)` | ASYNC | `test_upload_success`, `test_upload_invalid_mime`, `test_upload_too_large`, `test_upload_executable` | TestClient |

**NOVO (1 rota adicional):**

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 16 | `get_candidates(process_id, candidate_svc)` | ASYNC | `test_get_candidates_list`, `test_get_candidates_empty`, `test_get_candidates_sorted` | TestClient |

---

## 🟢 SERVICES — SYNC (34 métodos)

> ✅ Todos são `def`. Testes simples sem async.

### 1. auth_service.py (11 métodos SYNC)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 1 | `register(register_data)` | SYNC | `test_register_valid`, `test_register_duplicate`, `test_register_hash_argon2` |
| 2 | `login(email, password)` | SYNC | `test_login_valid`, `test_login_invalid_password`, `test_login_user_not_found` |
| 3 | `refresh_access_token(refresh_token)` | SYNC | `test_refresh_valid`, `test_refresh_expired`, `test_refresh_blacklisted` |
| 4 | `logout(refresh_token)` | SYNC | `test_logout_revokes_token`, `test_logout_already_revoked` |
| 5 | `_hash_password(password)` | SYNC | `test_hash_argon2_format`, `test_hash_different_each_time` |
| 6 | `_verify_password(password, hashed)` | SYNC | `test_verify_correct_password`, `test_verify_wrong_password` |
| 7 | `_create_token(data, expire_delta)` | SYNC | `test_create_token_format`, `test_create_token_claims` |
| 8 | `_verify_token(token)` | SYNC | `test_verify_valid_token`, `test_verify_expired_token`, `test_verify_invalid_signature` |
| 9 | `_hash_token(token)` | SYNC | `test_hash_token_consistent`, `test_hash_token_not_plaintext` |
| 10 | `_save_refresh_token(user_id, token_hash)` | SYNC | `test_save_refresh_token_in_db` |
| 11 | `_is_token_blacklisted(token_hash)` | SYNC | `test_blacklist_check_present`, `test_blacklist_check_absent` |

**NOVO (2 métodos adicionais):**

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 12 | `validate_email_format(email)` | SYNC | `test_validate_email_valid`, `test_validate_email_invalid` |
| 13 | `validate_password_strength(password)` | SYNC | `test_validate_password_strong`, `test_validate_password_weak` |

### 2. process_service.py (11 métodos SYNC)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 14 | `create_process(title, jd_text)` | SYNC | `test_create_process_valid`, `test_create_process_empty_title`, `test_create_process_empty_jd` |
| 15 | `list_processes(offset, limit)` | SYNC | `test_list_empty`, `test_list_paginated`, `test_list_limit_exceeded` |
| 16 | `get_process(process_id)` | SYNC | `test_get_process_found`, `test_get_process_not_found` |
| 17 | `update_status(process_id, new_status)` | SYNC | `test_update_status_valid_transition`, `test_update_status_invalid_transition` |
| 18 | `_is_valid_transition(current, new)` | SYNC | `test_valid_transitions`, `test_invalid_transitions` |
| 19 | `mark_files_uploaded(process_id)` | SYNC | `test_mark_files_uploaded` |
| 20 | `mark_processing(process_id)` | SYNC | `test_mark_processing` |
| 21 | `mark_completed(process_id)` | SYNC | `test_mark_completed` |
| 22 | `mark_failed(process_id, error_msg)` | SYNC | `test_mark_failed` |
| 23 | `cancel_process(process_id)` | SYNC | `test_cancel_process` |
| 24 | `delete_process(process_id)` | SYNC | `test_delete_process_cascade` |

**NOVO (1 método adicional):**

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 25 | `get_process_by_title(title)` | SYNC | `test_get_by_title_found`, `test_get_by_title_not_found` |

### 3. candidate_service.py (8 métodos SYNC)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 26 | `validate_file(file)` | SYNC | `test_validate_csv_valid`, `test_validate_executable_blocked`, `test_validate_size_exceeded`, `test_validate_mime_mismatch` |
| 27 | `save_file(process_id, candidate_name, file_content)` | SYNC | `test_save_file_success`, `test_save_file_directory_created` |
| 28 | `list_candidates(process_id)` | SYNC | `test_list_candidates_empty`, `test_list_candidates_sorted` |
| 29 | `get_candidate(candidate_id)` | SYNC | `test_get_candidate_found`, `test_get_candidate_not_found` |
| 30 | `delete_candidate(candidate)` | SYNC | `test_delete_candidate_from_db` |
| 31 | `get_file_path(candidate_id)` | SYNC | `test_get_file_path_exists`, `test_get_file_path_not_found` |
| 32 | `validate_csv_headers(csv_content)` | SYNC | `test_validate_headers_valid`, `test_validate_headers_missing` |
| 33 | `parse_csv_rows(csv_content)` | SYNC | `test_parse_csv_valid`, `test_parse_csv_corrupted` |

**NOVO (1 método adicional):**

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 34 | `count_candidates(process_id)` | SYNC | `test_count_candidates_zero`, `test_count_candidates_multiple` |

### 4. screening_service.py (4 métodos SYNC)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 35 | `run(process_id, db_session)` | SYNC | `test_run_screening_success`, `test_run_no_candidates`, `test_run_exception_handling` |
| 36 | `get_results(process_id, db_session)` | SYNC | `test_get_results_ranked`, `test_get_results_empty` |
| 37 | `_load_jd_criteria(jd_text)` | SYNC | `test_load_jd_criteria_parsing` |
| 38 | `_process_candidate(candidate, jd_criteria)` | SYNC | `test_process_candidate_scoring` |

### 5. report_service.py (8 métodos SYNC)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 39 | `export_csv(process_id, db_session)` | SYNC | `test_export_csv_valid`, `test_export_csv_not_completed`, `test_export_csv_empty_candidates` |
| 40 | `export_json(process_id, db_session)` | SYNC | `test_export_json_valid`, `test_export_json_metadata` |
| 41 | `export_txt(process_id, db_session)` | SYNC | `test_export_txt_valid`, `test_export_txt_narrative` |
| 42 | `_load_and_validate_process(process_id, db_session)` | SYNC | `test_validate_process_exists`, `test_validate_process_completed` |
| 43 | `_build_candidates_list(process, db_session)` | SYNC | `test_build_candidates_ranked`, `test_build_candidates_empty` |
| 44 | `_cleanup_tempfile(temp_path)` | SYNC | `test_cleanup_removes_file` |

**NOVO (2 métodos adicionais):**

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 45 | `export_pdf(process_id, db_session)` | SYNC | `test_export_pdf_valid`, `test_export_pdf_format` |
| 46 | `get_export_metadata(process_id, db_session)` | SYNC | `test_export_metadata_summary`, `test_export_metadata_timestamp` |

---

## 🟡 DEPENDENCIES — MIXED (10 métodos)

### get_current_user (1 ASYNC)

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 1 | `get_current_user(token)` | ASYNC | `test_get_current_user_valid`, `test_get_current_user_expired`, `test_get_current_user_missing_header` | TestClient |

**NOVO (1 adicional):**

| # | Método | Tipo | Testes | Ajustes |
|---|--------|------|--------|---------|
| 2 | `get_current_user_optional(token)` | ASYNC | `test_get_optional_user_with_token`, `test_get_optional_user_without_token` | TestClient |

### Outros Dependencies (9 SYNC)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 3 | `validate_process_id(process_id)` | SYNC | `test_validate_valid_uuid`, `test_validate_invalid_uuid` |
| 4 | `get_db()` | SYNC | `test_get_db_returns_session` |
| 5 | `get_nlp_model(request)` | SYNC | `test_get_nlp_model_loaded` |
| 6 | `get_process_service(db)` | SYNC | `test_get_process_service_instance` |
| 7 | `get_candidate_service(db, process_svc)` | SYNC | `test_get_candidate_service_instance` |
| 8 | `get_screening_service(nlp)` | SYNC | `test_get_screening_service_instance` |
| 9 | `get_report_service()` | SYNC | `test_get_report_service_instance` |
| 10 | `get_auth_service(db, settings)` | SYNC | `test_get_auth_service_instance` |
| 11 | `validate_content_type(content_type)` | SYNC | `test_validate_content_type_csv`, `test_validate_content_type_invalid` |

**NOVO (2 adicionais):**

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 12 | `validate_user_ownership(user_id, resource)` | SYNC | `test_ownership_valid`, `test_ownership_forbidden` |

---

## 🔵 UTILS — SYNC (18 métodos)

### validators.py (6 métodos)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 1 | `validate_uuid(uuid_str)` | SYNC | `test_uuid_valid`, `test_uuid_invalid_format`, `test_uuid_none` |
| 2 | `validate_mime_type(mime, allowed)` | SYNC | `test_mime_csv_valid`, `test_mime_executable_blocked`, `test_mime_mismatch` |
| 3 | `validate_file_size(size, max_bytes)` | SYNC | `test_size_valid`, `test_size_exceeded`, `test_size_zero` |
| 4 | `validate_email(email)` | SYNC | `test_email_valid`, `test_email_invalid_format` |
| 5 | `validate_password_requirements(password)` | SYNC | `test_password_min_length`, `test_password_no_special_char` |
| 6 | `sanitize_filename(filename)` | SYNC | `test_sanitize_removes_special_chars`, `test_sanitize_unicode` |

**NOVO (2 adicionais):**

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 7 | `validate_csv_structure(csv_content)` | SYNC | `test_csv_valid_structure`, `test_csv_missing_columns` |
| 8 | `validate_jd_text_length(jd_text)` | SYNC | `test_jd_min_length`, `test_jd_max_length` |

### errors.py (4 métodos)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 9 | `ValidationError.__init__` | SYNC | `test_validation_error_format` |
| 10 | `NotFoundError.__init__` | SYNC | `test_not_found_error_format` |
| 11 | `UnauthorizedError.__init__` | SYNC | `test_unauthorized_error_format` |
| 12 | `ConflictError.__init__` | SYNC | `test_conflict_error_format` |

**NOVO (2 adicionais):**

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 13 | `ForbiddenError.__init__` | SYNC | `test_forbidden_error_format` |
| 14 | `error_response_format(error)` | SYNC | `test_error_response_json_structure` |

### logging.py (2 métodos)

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 15 | `setup_logging(config)` | SYNC | `test_logging_initialized` |
| 16 | `mask_sensitive_data(log_msg)` | SYNC | `test_mask_password`, `test_mask_token`, `test_mask_email` |

**NOVO (2 adicionais):**

| # | Método | Tipo | Testes |
|---|--------|------|--------|
| 17 | `log_request_metadata(request)` | SYNC | `test_log_request_id`, `test_log_user_agent` |
| 18 | `log_response_time(duration)` | SYNC | `test_log_performance_slow`, `test_log_performance_fast` |

---

## 🟣 MODELS — SYNC (12 métodos)

### ORM Models (6 métodos)

| # | Modelo | Método | Tipo | Testes |
|---|--------|--------|------|--------|
| 1 | User | `__init__` | SYNC | `test_user_creation`, `test_user_defaults` |
| 2 | Process | `__init__` | SYNC | `test_process_creation`, `test_process_status_enum` |
| 3 | Candidate | `__init__` | SYNC | `test_candidate_creation`, `test_candidate_file_path` |
| 4 | Result | `__init__` | SYNC | `test_result_creation`, `test_result_scoring` |
| 5 | RefreshToken | `__init__` | SYNC | `test_refresh_token_creation`, `test_refresh_token_expiration` |
| 6 | TokenBlacklist | `__init__` | SYNC | `test_blacklist_creation`, `test_blacklist_timestamp` |

**NOVO (3 adicionais):**

| # | Modelo | Método | Tipo | Testes |
|---|--------|--------|------|--------|
| 7 | ProcessingError | `__init__` | SYNC | `test_processing_error_creation`, `test_error_message` |
| 8 | ProcessStatus | `__init__` (enum) | SYNC | `test_status_enum_values`, `test_status_transitions` |
| 9 | AuditLog | `__init__` | SYNC | `test_audit_log_creation`, `test_audit_log_user_action` |

### Pydantic Schemas (3 métodos)

| # | Schema | Método | Tipo | Testes |
|---|--------|--------|------|--------|
| 10 | UserRegister | `validate()` | SYNC | `test_user_register_valid`, `test_user_register_email_format` |
| 11 | ProcessCreate | `validate()` | SYNC | `test_process_create_valid`, `test_process_create_empty_fields` |
| 12 | UploadFile | `validate()` | SYNC | `test_upload_file_valid`, `test_upload_file_size_exceeded` |

---

## 📋 Tabela Consolidada — Ajustes Necessários

| Tipo | Fixture | Como Testar | Exemplo |
|------|---------|-------------|---------|
| **ROUTES** | `client` | `client.get(...)` | `response = client.get("/api/processes")` |
| **SERVICES** | `service_instance` | `service.method(...)` | `user = auth_service.register(data)` |
| **DEPENDENCIES** | `test_db, client` | Via route test | Via `client.get(...)` com headers |
| **UTILS** | Nenhuma | Import direto | `validate_uuid("...")` |
| **MODELS** | Nenhuma | Import direto | `User(email="test@example.com")` |

---

## 🛠️ Conftest.py Mínimo Necessário

```python
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.api.db.database import get_db, Base, engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ✅ In-memory test database
@pytest.fixture(scope="function")
def test_db():
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)

# ✅ FastAPI TestClient (executa async routes sincronamente)
@pytest.fixture
def client(test_db):
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

# ✅ Service fixtures (sem async)
@pytest.fixture
def auth_service(test_db):
    from backend.api.services.auth_service import AuthService
    from backend.api.config import get_settings
    return AuthService(db=test_db, settings=get_settings())

@pytest.fixture
def process_service(test_db):
    from backend.api.services.process_service import ProcessService
    return ProcessService(db=test_db)

@pytest.fixture
def candidate_service(test_db, process_service):
    from backend.api.services.candidate_service import CandidateService
    return CandidateService(db=test_db, process_service=process_service)

@pytest.fixture
def screening_service():
    from backend.api.services.screening_service import ScreeningService
    import spacy
    nlp = spacy.load("en_core_web_sm")
    return ScreeningService(nlp=nlp)

@pytest.fixture
def report_service():
    from backend.api.services.report_service import ReportService
    return ReportService()

# ✅ Test data fixtures
@pytest.fixture
def test_user(client):
    from backend.api.models.schemas import UserRegister
    response = client.post("/api/auth/register", json={
        "email": "test@example.com",
        "password": "TestPassword123!"
    })
    return response.json()

@pytest.fixture
def auth_header(test_user):
    # Login para obter token
    # return {"Authorization": f"Bearer {token}"}
    pass

@pytest.fixture
def test_process(client, auth_header):
    response = client.post("/api/processes", 
        json={"title": "Test Job", "jd_text": "5+ years Python"},
        headers=auth_header
    )
    return response.json()
```

---

## ✅ Checklist de Implementação

- [ ] Criar `conftest.py` com todas as fixtures
- [ ] Implementar **14 tests para ROUTES** (4 auth + 3 processes + 3 results + 2 upload + 2 novos)
- [ ] Implementar **35 tests para SERVICES** (11 auth + 11 process + 8 candidate + 4 screening + 8 report)
- [ ] Implementar **10 tests para DEPENDENCIES**
- [ ] Implementar **18 tests para UTILS**
- [ ] Implementar **12 tests para MODELS**
- [ ] Implementar **30+ INTEGRATION tests** (full flows)
- [ ] Implementar **20+ SECURITY tests** (JWT, file upload, SQL injection, XSS)
- [ ] Executar: `pytest backend/api/tests/ -v --cov=backend/api --cov-fail-under=75`
- [ ] Validar **75%+ code coverage**

---

## 📈 Progresso Esperado

| Fase | Testes | Coverage | Tempo |
|------|--------|----------|-------|
| **Fase 1** (Services) | 35 | 50% | 1-2 dias |
| **Fase 2** (Routes) | 14 | 60% | 1-2 dias |
| **Fase 3** (Integration) | 30 | 70% | 2-3 dias |
| **Fase 4** (Security) | 20 | 75% | 1-2 dias |
| **TOTAL** | **120** | **75%** | **~7 dias** |

---

**Sprint 7 — Ready for Full Test Implementation 🚀**
