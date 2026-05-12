# Sprint 7 — Test Plan (Testes e Segurança) — COMPLETO

**Status:** 📋 PLANEJAMENTO | **Data:** Maio 2026

**Objetivo:** Atingir cobertura mínima 70%, validar todos os sprints anteriores (0-6), security & performance

---

## 1. Arquitetura de Testes

### Estrutura de pastas
```
backend/api/tests/
├── __init__.py
├── conftest.py                  # Fixtures compartilhadas
├── unit/
│   ├── test_validators.py       # Utils validators
│   ├── test_errors.py           # Custom exceptions
│   ├── services/
│   │   ├── test_auth_service.py
│   │   ├── test_process_service.py
│   │   ├── test_candidate_service.py
│   │   ├── test_screening_service.py
│   │   ├── test_report_service.py
│   └── routes/
│       ├── test_auth_routes.py
│       ├── test_processes_routes.py
│       ├── test_upload_routes.py
│       ├── test_results_routes.py
├── integration/
│   ├── test_full_flow_screening.py
│   ├── test_full_flow_export.py
│   ├── test_state_transitions.py
│   ├── test_database_integrity.py
├── security/
│   ├── test_jwt_validation.py
│   ├── test_file_upload_security.py
│   ├── test_path_traversal.py
│   ├── test_sql_injection.py
│   ├── test_xss_prevention.py
└── load/
    └── test_performance.py
```

---

## 2. Unit Tests — Services (Backend Logic)

### 2.1 test_auth_service.py (6 testes)

**Dependências:** User ORM, JWT secret, password hashing (Argon2)

```python
# Testes a implementar:

def test_register_user_success():
    """Registra novo utilizador, hash de password, retorna User schema"""
    # Given: email, password válidos
    # When: auth_service.register(email, password)
    # Then: User criado no DB, password é hash (nunca plaintext)

def test_register_user_duplicate_email():
    """Tenta registar com email existente"""
    # Given: email já existe
    # When: auth_service.register(email, password)
    # Then: ConflictError (400), sem duplicates no DB

def test_register_user_invalid_email():
    """Email inválido não passa validação"""
    # Given: email = "invalid-email"
    # When: auth_service.register(invalid_email, password)
    # Then: ValidationError (400)

def test_login_success():
    """Login correto retorna JWT válido"""
    # Given: user registado, credenciais corretas
    # When: auth_service.login(email, password)
    # Then: JWT retornado, válido, contém user_id

def test_login_invalid_password():
    """Password errada retorna 401"""
    # Given: user exists, password incorreto
    # When: auth_service.login(email, wrong_password)
    # Then: UnauthorizedError (401), sem JWT

def test_get_current_user_from_jwt():
    """Extrai user do JWT válido"""
    # Given: JWT válido com user_id
    # When: auth_service.get_current_user(token)
    # Then: User object retornado, matching user_id
```

**Status codes a testar:** 200 (success), 400 (validation), 401 (auth), 409 (conflict)

---

### 2.2 test_process_service.py (8 testes)

**Dependências:** Process ORM, ProcessStatus enum, user ownership

```python
def test_create_process_success():
    """Cria novo processo para user autenticado"""
    # Given: user autenticado, jd_text válido
    # When: process_service.create(user_id, jd_text, title)
    # Then: Process criado em DB, status = CREATED, owner = user_id

def test_create_process_missing_jd():
    """JD ausente não é aceito"""
    # Given: jd_text = None ou vazio
    # When: process_service.create(user_id, None, title)
    # Then: ValidationError (400)

def test_get_process_success():
    """Retorna processo do user"""
    # Given: process existe, user é owner
    # When: process_service.get(process_id, user_id)
    # Then: Process object retornado

def test_get_process_not_found():
    """Processo inexistente retorna 404"""
    # Given: process_id não existe
    # When: process_service.get(invalid_id, user_id)
    # Then: NotFoundError (404)

def test_get_process_not_owned():
    """User não pode ver processos de outro user"""
    # Given: process existe, owner = user_a, requester = user_b
    # When: process_service.get(process_id, user_b_id)
    # Then: ForbiddenError (403), access denied

def test_list_processes_for_user():
    """Lista todos os processos do user"""
    # Given: user tem 3 processos no DB
    # When: process_service.list_by_user(user_id)
    # Then: Retorna list[Process], todos pertencentes a user

def test_update_process_status():
    """Atualiza status de processo"""
    # Given: process existe, status = CREATED
    # When: process_service.update_status(process_id, PROCESSING)
    # Then: process.status = PROCESSING, updated_at = now

def test_delete_process():
    """Apaga processo e cascade deletes (candidates, results)"""
    # Given: process existe com 5 candidatos e 5 resultados
    # When: process_service.delete(process_id)
    # Then: process, candidates, results all deleted from DB
```

**Status codes a testar:** 200, 400, 403, 404

---

### 2.3 test_candidate_service.py (7 testes)

**Dependências:** Candidate ORM, file upload handling

```python
def test_create_candidate_from_file_pdf():
    """Faz upload de PDF, cria Candidate no DB"""
    # Given: file.pdf válido, process_id existe
    # When: candidate_service.create_from_upload(process_id, file)
    # Then: Candidate criado, file_path armazenado, file parsed

def test_create_candidate_invalid_mime_type():
    """Rejeita ficheiros não-PDF/DOCX/TXT"""
    # Given: file.exe (ou outro tipo)
    # When: candidate_service.create_from_upload(process_id, file)
    # Then: ValidationError (400), "Invalid file type"

def test_create_candidate_file_too_large():
    """Rejeita ficheiros > MAX_FILE_SIZE (ex: 10MB)"""
    # Given: file.pdf com 50MB
    # When: candidate_service.create_from_upload(process_id, file)
    # Then: ValidationError (413), "File too large"

def test_get_candidate():
    """Retorna candidate por ID"""
    # Given: candidate existe
    # When: candidate_service.get(candidate_id)
    # Then: Candidate object retornado

def test_list_candidates_by_process():
    """Lista todos os candidatos de um processo"""
    # Given: process tem 10 candidatos
    # When: candidate_service.list_by_process(process_id)
    # Then: Retorna list[Candidate], len = 10

def test_delete_candidate():
    """Apaga candidate e cascade deletes results"""
    # Given: candidate existe com 1 result
    # When: candidate_service.delete(candidate_id)
    # Then: candidate e seu result deletados

def test_candidate_ownership_validation():
    """Candidate pertence a um processo de um user específico"""
    # Given: candidate belongs to process owned by user_a
    # When: user_b tenta acessar candidate
    # Then: ForbiddenError (403)
```

**Status codes a testar:** 200, 400, 403, 404, 413

---

### 2.4 test_screening_service.py (5 testes)

**Dependências:** ScreeningService, ReportGenerator (v1.0 pipeline), process state

```python
def test_run_screening_success():
    """Executa screening de um processo com candidatos"""
    # Given: process.status = CREATED, 5 candidates
    # When: screening_service.run_screening(process_id)
    # Then: Results criados (5), process.status = COMPLETED, scores ordenados DESC

def test_run_screening_no_candidates():
    """Screening sem candidatos retorna erro ou lista vazia"""
    # Given: process exists, 0 candidates
    # When: screening_service.run_screening(process_id)
    # Then: Handling gracioso (error ou empty results)

def test_run_screening_state_transition():
    """Valida transições de estado (CREATED → PROCESSING → COMPLETED)"""
    # Given: process.status = CREATED
    # When: screening_service.run_screening(process_id)
    # Then: status transitions: CREATED → PROCESSING → COMPLETED

def test_get_results_processing():
    """GET /results retorna 202 enquanto screening está a correr"""
    # Given: screening em background
    # When: screening_service.get_results(process_id) durante processing
    # Then: Retorna 202 Accepted, status = "processing"

def test_get_results_completed():
    """GET /results retorna 200 com results quando screening completo"""
    # Given: screening completado
    # When: screening_service.get_results(process_id)
    # Then: Retorna 200, list[Result] ordenados por score DESC
```

**Status codes a testar:** 200, 202, 400, 404

---

### 2.5 test_report_service.py (8 testes)

**Dependências:** ReportService, ReportGenerator, tempfile

```python
def test_export_csv_success():
    """Exporta CSV com resultados ranqueados"""
    # Given: process.status = COMPLETED, 5 candidatos
    # When: report_service.export_csv(process_id, db_session)
    # Then: Retorna (bytes, filename), bytes é CSV válido com 5 linhas

def test_export_csv_process_not_found():
    """Process inexistente retorna erro"""
    # Given: process_id não existe
    # When: report_service.export_csv(invalid_id, db_session)
    # Then: NotFoundError, detail menciona "Process ... not found"

def test_export_csv_process_not_completed():
    """Process em estado PROCESSING não pode ser exportado"""
    # Given: process.status = PROCESSING
    # When: report_service.export_csv(process_id, db_session)
    # Then: ValidationError (400), detail menciona "must be completed"

def test_export_json_success():
    """Exporta JSON com metadata e candidates"""
    # Given: process.status = COMPLETED
    # When: report_service.export_json(process_id, db_session)
    # Then: Retorna (bytes, filename), bytes é JSON válido com structure correto

def test_export_txt_success():
    """Exporta TXT narrativo legível"""
    # Given: process.status = COMPLETED
    # When: report_service.export_txt(process_id, db_session)
    # Then: Retorna (bytes, filename), bytes é TXT com seções (summary, ranking)

def test_tempfile_cleanup():
    """Temp directory é deletado após export"""
    # Given: export_csv(process_id, db_session)
    # When: context manager sai
    # Then: temp directory não existe mais no filesystem

def test_csv_format_validation():
    """CSV gerado tem headers corretos e dados formatados"""
    # Given: export_csv result
    # When: parsear CSV
    # Then: headers = [Rank, Candidate, Total Score, ...], 5 linhas de dados

def test_multiple_exports_same_process():
    """Múltiplas exportações do mesmo processo funcionam"""
    # Given: process exists
    # When: export_csv 3x, export_json 1x, export_txt 1x
    # Then: Todos retornam bytes válidos (sem conflitos de tempfile)
```

**Status codes a testar:** 200, 400, 404

---

## 3. Unit Tests — Routes (HTTP Layer)

### 3.1 test_auth_routes.py (6 testes)

```python
def test_post_register_success():
    """POST /api/auth/register com credenciais válidas"""
    # Given: body = {email, password}
    # When: POST /api/auth/register
    # Then: 200 OK, response = {user_id, email}

def test_post_register_invalid_email():
    """POST /api/auth/register com email inválido"""
    # Given: body = {email: "not-email", password: "..."}
    # When: POST /api/auth/register
    # Then: 400 Bad Request, error message

def test_post_login_success():
    """POST /api/auth/login com credenciais corretas"""
    # Given: user exists, body = {email, password correto}
    # When: POST /api/auth/login
    # Then: 200 OK, response = {access_token: "jwt", token_type: "bearer"}

def test_post_login_invalid_credentials():
    """POST /api/auth/login com password errada"""
    # Given: body = {email, password incorreto}
    # When: POST /api/auth/login
    # Then: 401 Unauthorized

def test_post_login_user_not_found():
    """POST /api/auth/login com email não registado"""
    # Given: body = {email: "nonexistent@test.com", password}
    # When: POST /api/auth/login
    # Then: 401 Unauthorized (não revela se email existe ou não)

def test_auth_header_required():
    """Endpoints protegidos retornam 401 sem Authorization header"""
    # Given: GET /api/processes (protected)
    # When: sem Authorization header
    # Then: 401 Unauthorized
```

**HTTP methods a testar:** POST (register, login), GET (protected)

---

### 3.2 test_processes_routes.py (7 testes)

```python
def test_post_create_process():
    """POST /api/processes com JD válido"""
    # Given: Authorization header com JWT, body = {jd_text, title}
    # When: POST /api/processes
    # Then: 201 Created, response = {process_id, status: "created"}

def test_post_create_process_unauthorized():
    """POST /api/processes sem JWT"""
    # Given: sem Authorization header
    # When: POST /api/processes
    # Then: 401 Unauthorized

def test_get_list_processes():
    """GET /api/processes lista processos do user autenticado"""
    # Given: user tem 3 processos, JWT válido
    # When: GET /api/processes
    # Then: 200 OK, response = list[Process], len = 3

def test_get_process_by_id():
    """GET /api/processes/{id}"""
    # Given: process exists, user é owner
    # When: GET /api/processes/{process_id}
    # Then: 200 OK, response = Process object

def test_get_process_not_found():
    """GET /api/processes/{id} inexistente"""
    # Given: process_id não existe
    # When: GET /api/processes/{invalid_id}
    # Then: 404 Not Found

def test_get_process_forbidden():
    """GET /api/processes/{id} de outro user"""
    # Given: process pertence a outro user
    # When: GET /api/processes/{process_id} (outro user)
    # Then: 403 Forbidden

def test_delete_process():
    """DELETE /api/processes/{id}"""
    # Given: process exists, user é owner
    # When: DELETE /api/processes/{process_id}
    # Then: 204 No Content, process deletado do DB
```

**HTTP methods a testar:** POST, GET, DELETE

---

### 3.3 test_upload_routes.py (6 testes)

```python
def test_post_upload_pdf():
    """POST /api/processes/{id}/upload com ficheiro PDF"""
    # Given: process exists, file = test.pdf (válido)
    # When: POST multipart/form-data
    # Then: 201 Created, response = {candidate_id, filename}

def test_post_upload_docx():
    """POST /api/processes/{id}/upload com ficheiro DOCX"""
    # Given: process exists, file = test.docx
    # When: POST multipart/form-data
    # Then: 201 Created

def test_post_upload_invalid_type():
    """POST /api/processes/{id}/upload com ficheiro .exe"""
    # Given: file.exe
    # When: POST multipart/form-data
    # Then: 400 Bad Request, "Invalid file type"

def test_post_upload_file_too_large():
    """POST /api/processes/{id}/upload com ficheiro > MAX_SIZE"""
    # Given: file.pdf, size = 100MB (max = 10MB)
    # When: POST multipart/form-data
    # Then: 413 Payload Too Large

def test_post_upload_process_not_found():
    """POST /api/processes/{invalid_id}/upload"""
    # Given: process_id não existe
    # When: POST multipart/form-data
    # Then: 404 Not Found

def test_post_upload_multiple_files():
    """POST múltiplos ficheiros num mesmo processo"""
    # Given: process exists
    # When: POST 5 ficheiros sequencialmente
    # Then: 5 candidates criados, todos com process_id correto
```

**HTTP methods a testar:** POST (multipart/form-data)

---

### 3.4 test_results_routes.py (10 testes)

```python
def test_post_run_screening():
    """POST /api/processes/{id}/run dispara screening em background"""
    # Given: process exists, status = CREATED, 3 candidates
    # When: POST /api/processes/{id}/run
    # Then: 200 OK, response = {status: "queued"}, process.status muda para PROCESSING

def test_post_run_screening_no_candidates():
    """POST /api/processes/{id}/run sem candidatos"""
    # Given: process exists, 0 candidates
    # When: POST /api/processes/{id}/run
    # Then: 400 Bad Request, "No candidates to screen"

def test_get_results_processing():
    """GET /api/processes/{id}/results enquanto screening corre"""
    # Given: screening em background
    # When: GET /api/processes/{id}/results
    # Then: 202 Accepted, response = {status: "processing"}

def test_get_results_completed():
    """GET /api/processes/{id}/results quando screening completado"""
    # Given: screening finalizado, 3 candidates, results created
    # When: GET /api/processes/{id}/results
    # Then: 200 OK, response = list[Result] com 3 items, ordenados por score DESC

def test_get_results_not_found():
    """GET /api/processes/{invalid_id}/results"""
    # Given: process_id não existe
    # When: GET /api/processes/{invalid_id}/results
    # Then: 404 Not Found

def test_get_export_csv_success():
    """GET /api/processes/{id}/export/csv retorna ficheiro"""
    # Given: process.status = COMPLETED
    # When: GET /api/processes/{id}/export/csv
    # Then: 200 OK, Content-Type = text/csv, Content-Disposition = attachment, body = CSV bytes

def test_get_export_csv_not_completed():
    """GET /api/processes/{id}/export/csv de processo em PROCESSING"""
    # Given: process.status = PROCESSING
    # When: GET /api/processes/{id}/export/csv
    # Then: 400 Bad Request

def test_get_export_json_success():
    """GET /api/processes/{id}/export/json"""
    # Given: process.status = COMPLETED
    # When: GET /api/processes/{id}/export/json
    # Then: 200 OK, Content-Type = application/json, body = JSON bytes

def test_get_export_txt_success():
    """GET /api/processes/{id}/export/txt"""
    # Given: process.status = COMPLETED
    # When: GET /api/processes/{id}/export/txt
    # Then: 200 OK, Content-Type = text/plain, body = TXT bytes

def test_get_export_unauthorized():
    """GET /api/processes/{id}/export/csv sem JWT"""
    # Given: sem Authorization header
    # When: GET /api/processes/{id}/export/csv
    # Then: 401 Unauthorized
```

**HTTP methods a testar:** POST, GET

---

## 4. Unit Tests — Utilities

### 4.1 test_validators.py (5 testes)

```python
def test_validate_process_id_valid_uuid():
    """UUID válido passa validação"""
    # Given: process_id = "550e8400-e29b-41d4-a716-446655440000"
    # When: validate_process_id(process_id)
    # Then: Retorna process_id (sem erro)

def test_validate_process_id_invalid_uuid():
    """UUID inválido levanta ValidationError"""
    # Given: process_id = "invalid-uuid"
    # When: validate_process_id(process_id)
    # Then: ValidationError (400)

def test_validate_file_mime_type_pdf():
    """Valida ficheiro PDF"""
    # Given: file.pdf
    # When: validate_file_mime_type(file, ["application/pdf"])
    # Then: True (válido)

def test_validate_file_mime_type_invalid():
    """Rejeita ficheiro não PDF/DOCX/TXT"""
    # Given: file.exe
    # When: validate_file_mime_type(file, allowed_types)
    # Then: ValidationError

def test_validate_file_size():
    """Valida tamanho de ficheiro"""
    # Given: file.pdf, size = 5MB, max = 10MB
    # When: validate_file_size(file, 10 * 1024 * 1024)
    # Then: True (válido)
```

---

### 4.2 test_errors.py (4 testes)

```python
def test_validation_error_to_json():
    """ValidationError converte para JSON (400)"""
    # Given: ValidationError("Invalid input")
    # When: catch e retornar em response
    # Then: JSON = {detail: "Invalid input"}, status_code = 400

def test_not_found_error_to_json():
    """NotFoundError converte para JSON (404)"""
    # Given: NotFoundError("Resource not found")
    # When: catch e retornar em response
    # Then: JSON = {detail: "Resource not found"}, status_code = 404

def test_unauthorized_error_to_json():
    """UnauthorizedError converte para JSON (401)"""
    # Given: UnauthorizedError("Invalid token")
    # When: catch e retornar em response
    # Then: JSON = {detail: "Invalid token"}, status_code = 401

def test_forbidden_error_to_json():
    """ForbiddenError converte para JSON (403)"""
    # Given: ForbiddenError("Access denied")
    # When: catch e retornar em response
    # Then: JSON = {detail: "Access denied"}, status_code = 403
```

---

## 5. Integration Tests — End-to-End Flows

### 5.1 test_full_flow_screening.py (3 testes)

```python
def test_flow_register_to_results():
    """Full flow: register → create process → upload 3 CVs → run screening → get results"""
    # 1. POST /api/auth/register → user criado
    # 2. POST /api/processes → process criado (status=CREATED)
    # 3. POST /api/processes/{id}/upload × 3 → 3 candidates criados
    # 4. POST /api/processes/{id}/run → screening inicia
    # 5. POLL GET /api/processes/{id}/results → 202 (processing) → 200 (completed)
    # 6. Verificar: results.count = 3, todos com scores, ordenados DESC

def test_flow_multiple_processes_same_user():
    """User cria 2 processos, faz upload em ambos, runs screening em ambos"""
    # 1. register → user
    # 2. POST /api/processes × 2 → 2 processes
    # 3. upload CVs em ambos
    # 4. run screening em ambos
    # 5. get results ambos → todos completos, scores diferentes por JD

def test_flow_concurrent_uploads():
    """2 users fazem uploads simultaneamente no mesmo processo (se allowed)"""
    # Nota: dependente de regras de ownership; teste pode ser skip se não-allowed
    # Given: 2 users, 1 process
    # When: upload simultaneamente
    # Then: handling correto (serial ou parallel sem corruptions)
```

---

### 5.2 test_full_flow_export.py (3 testes)

```python
def test_flow_screening_to_csv_export():
    """Full flow: screening completado → export CSV → validar formato"""
    # 1. Setup: process com 5 candidates, screening completado
    # 2. GET /api/processes/{id}/export/csv
    # 3. Validar: 200 OK, CSV bytes
    # 4. Parse CSV: headers corretos, 5 linhas de dados
    # 5. Verificar: scores ordenados DESC

def test_flow_export_all_formats():
    """Export em CSV, JSON, TXT; validar todos"""
    # 1. Setup: process completado
    # 2. GET /api/processes/{id}/export/csv → validar
    # 3. GET /api/processes/{id}/export/json → validar JSON structure
    # 4. GET /api/processes/{id}/export/txt → validar TXT format
    # 5. Verificar: todos têm mesmos dados (scores, candidates, order)

def test_flow_export_large_dataset():
    """Export com 1000 candidates; verificar performance e memory"""
    # 1. Setup: 1000 candidates, screening completado
    # 2. GET /api/processes/{id}/export/csv
    # 3. Validar: 200 OK, < 5 segundos, memory < 100MB
    # 4. Validar: CSV é válido (não corrupted)
```

---

### 5.3 test_state_transitions.py (4 testes)

```python
def test_process_status_transitions():
    """Valida máquina de estados: CREATED → PROCESSING → COMPLETED"""
    # 1. POST /api/processes → status = CREATED ✓
    # 2. POST /api/processes/{id}/run → status muda para PROCESSING ✓
    # 3. [wait for background job] → status = COMPLETED ✓
    # 4. Verificar: transitions são sequenciais, sem saltos

def test_process_status_failed():
    """Processo em PROCESSING levanta erro → status = FAILED"""
    # 1. POST /api/processes → CREATED
    # 2. POST /api/processes/{id}/run → PROCESSING
    # 3. [simulate error no screening] → status = FAILED
    # 4. GET /api/processes/{id} → status = FAILED, error_message presente

def test_prevent_double_run():
    """POST /api/processes/{id}/run enquanto já está PROCESSING retorna erro"""
    # 1. POST /run → PROCESSING
    # 2. POST /run novamente → 400 Bad Request, "Already processing"

def test_prevent_export_before_completion():
    """GET /api/processes/{id}/export/csv em PROCESSING retorna 400"""
    # 1. POST /run → PROCESSING
    # 2. GET /export/csv → 400 Bad Request, "Process must be completed"
    # 3. [wait for completion]
    # 4. GET /export/csv → 200 OK, CSV
```

---

### 5.4 test_database_integrity.py (4 testes)

```python
def test_cascade_delete_process():
    """Apagar processo apaga candidates e results (cascade)"""
    # 1. Setup: process com 5 candidates, cada com 1 result (5 total)
    # 2. DELETE /api/processes/{id}
    # 3. Query: candidates.count = 0, results.count = 0, process.count = 0
    # 4. Verificar: não há orphan records no DB

def test_foreign_key_constraints():
    """Não é possível inserir result sem candidate válido"""
    # 1. Tentar INSERT result com candidate_id inexistente
    # 2. DB levanta constraint violation (ou app validação)
    # 3. Transação é rolled back

def test_unique_constraints():
    """Email de user é unique; não há duplicates"""
    # 1. Register user1 → email = "test@example.com"
    # 2. Tentar register user2 → email = "test@example.com"
    # 3. 409 Conflict ou DB constraint violation

def test_data_consistency_after_concurrent_ops():
    """2 users fazem operations concurrentemente; DB fica consistente"""
    # 1. User A: register
    # 2. User B: register (simultaneamente)
    # 3. User A: create process + upload
    # 4. User B: create process + upload (simultaneamente)
    # 5. Verificar: 2 processes, 4 candidates, todas as records presentes, nenhuma duplicada
```

---

## 6. Security Tests

### 6.1 test_jwt_validation.py (5 testes)

```python
def test_expired_jwt():
    """JWT expirado é rejeitado"""
    # Given: JWT com exp = now - 1 hour
    # When: GET /api/processes com Authorization: Bearer <expired_token>
    # Then: 401 Unauthorized, "Token expired"

def test_invalid_jwt_signature():
    """JWT com signature inválida é rejeitado"""
    # Given: JWT válido mas tampered (última letra alterada)
    # When: GET /api/processes com Authorization: Bearer <tampered_token>
    # Then: 401 Unauthorized, "Invalid token"

def test_missing_authorization_header():
    """GET /api/processes sem Authorization header"""
    # Given: GET /api/processes (protected), sem header
    # When: Request
    # Then: 401 Unauthorized

def test_malformed_authorization_header():
    """Authorization header mal-formado (não "Bearer <token>")"""
    # Given: Authorization: "InvalidPrefix <token>"
    # When: GET /api/processes
    # Then: 401 Unauthorized

def test_jwt_user_id_validation():
    """JWT com user_id inexistente é rejeitado"""
    # Given: JWT válido com user_id = 99999 (não existe)
    # When: GET /api/processes
    # Then: 401 Unauthorized (user not found)
```

---

### 6.2 test_file_upload_security.py (5 testes)

```python
def test_block_executable_files():
    """Ficheiros executáveis (.exe, .sh, .bat) são bloqueados"""
    # Given: file.exe
    # When: POST /api/processes/{id}/upload
    # Then: 400 Bad Request, "Invalid file type"

def test_block_script_files():
    """Ficheiros script (.js, .py, .rb) são bloqueados"""
    # Given: file.js
    # When: POST /api/processes/{id}/upload
    # Then: 400 Bad Request

def test_mime_type_mismatch():
    """Ficheiro com MIME type falso (ex: .exe renomeado para .pdf) é detectado"""
    # Given: file = test.exe, mas Content-Type = application/pdf
    # When: POST /api/processes/{id}/upload
    # Then: 400 Bad Request (validação mime magic)

def test_file_size_limit():
    """Ficheiros > 10MB são rejeitados"""
    # Given: file.pdf, size = 100MB
    # When: POST /api/processes/{id}/upload
    # Then: 413 Payload Too Large

def test_no_path_traversal_in_filename():
    """Filename com ../ não permite path traversal"""
    # Given: filename = "../../etc/passwd"
    # When: POST /api/processes/{id}/upload
    # Then: Filename é sanitized (sem ../)
```

---

### 6.3 test_path_traversal.py (3 testes)

```python
def test_get_process_path_traversal():
    """GET /api/processes/../../admin retorna 400 ou 404 (não revela estrutura)"""
    # Given: GET /api/processes/../../admin
    # When: Request
    # Then: 400 Bad Request (UUID validation falha) ou 404

def test_download_file_path_traversal():
    """Não é possível fazer download de ficheiros fora do storage directory"""
    # Given: GET /api/download?file=../../etc/passwd
    # When: Request
    # Then: 400 Bad Request ou 403 Forbidden (se endpoint existe)

def test_sql_injection_in_process_id():
    """GET /api/processes/'; DROP TABLE processes; -- retorna erro"""
    # Given: process_id = "'; DROP TABLE processes; --"
    # When: GET /api/processes/{process_id}
    # Then: 400 Bad Request (UUID validation falha), DB não é afetada (parameterized queries)
```

---

### 6.4 test_sql_injection.py (3 testes)

```python
def test_sql_injection_in_email():
    """POST /api/auth/login com SQL injection em email"""
    # Given: email = "admin' OR '1'='1"
    # When: POST /api/auth/login
    # Then: 401 Unauthorized (parameterized queries, sem SQL execution)

def test_sql_injection_in_jd_text():
    """POST /api/processes com SQL injection em jd_text"""
    # Given: jd_text = "'; DROP TABLE processes; --"
    # When: POST /api/processes
    # Then: 201 Created (texto armazenado como-é, não executado)
    # Verificar: process.jd_text contém string literal

def test_parameterized_queries_everywhere():
    """Todas as queries usam parameterized (?) ou ORM, nunca f-strings"""
    # [Code review] Verificar backend/api/services/ e backend/api/routes/
    # Nenhum .query("SELECT ... WHERE id = " + str(id))
    # Todos usam .query(...).filter(Model.id == id)
```

---

### 6.5 test_xss_prevention.py (3 testes)

```python
def test_response_headers_security():
    """Response headers contêm X-Content-Type-Options, X-Frame-Options, etc"""
    # Given: GET /api/processes
    # When: Response
    # Then: Headers contêm:
    #   - X-Content-Type-Options: nosniff
    #   - X-Frame-Options: DENY
    #   - X-XSS-Protection: 1; mode=block

def test_json_escaping():
    """JSON responses escapam caracteres especiais (< > & " ')"""
    # Given: candidate.name = '<script>alert("xss")</script>'
    # When: GET /api/processes/{id}/results → response JSON
    # Then: JSON contém escaped version, nunca raw <script> tag

def test_csv_export_no_formula_injection():
    """CSV export escapa caracteres que podem ser fórmulas (=, +, -, @)"""
    # Given: candidate.name = "=1+1"
    # When: GET /api/processes/{id}/export/csv
    # Then: CSV contém "=1+1" como texto literal, não formula
```

---

## 7. Load Tests (Optional pero recomendado)

### 7.1 test_performance.py (4 testes)

```python
def test_export_csv_1000_candidates():
    """Export CSV com 1000 candidates, validar tempo e memória"""
    # Given: process com 1000 candidates, screening completado
    # When: GET /api/processes/{id}/export/csv
    # Then: 
    #   - Status: 200 OK
    #   - Time: < 5 segundos
    #   - Memory: < 200MB (streaming, não buffering all)
    #   - File size: ~ 100-500 KB (dependendo de dados)

def test_concurrent_export_requests():
    """100 concurrent requests para export (multiplas processes)"""
    # Given: 10 processes, cada com 100 candidates
    # When: 100 concurrent GET /api/processes/{id}/export/csv (10x cada)
    # Then:
    #   - Todos retornam 200 OK
    #   - Total time: < 30 segundos
    #   - No timeouts ou 500 errors
    #   - Server memory: < 500MB

def test_large_file_upload():
    """Upload de ficheiro PDF 9.9MB (perto do limit de 10MB)"""
    # Given: file.pdf, size = 9.9MB
    # When: POST /api/processes/{id}/upload
    # Then:
    #   - Status: 201 Created
    #   - Time: < 10 segundos
    #   - Server não fica lento após upload

def test_database_query_performance():
    """Query de results com 10000+ registos, verificar index usage"""
    # Given: 100 processes, cada com 100 candidates (10000 results total)
    # When: GET /api/processes/{id}/results (fetch all)
    # Then:
    #   - Time: < 500ms (com indexes)
    #   - EXPLAIN ANALYZE mostra index usage (não full table scan)
```

---

## 8. conftest.py — Fixtures Compartilhadas

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="function")
def test_db():
    """In-memory SQLite database para testes"""
    engine = create_engine("sqlite:///:memory:")
    # Create tables
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture(scope="function")
def client(test_db):
    """FastAPI TestClient com database injetado"""
    from backend.api.main import app
    
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

@pytest.fixture
def test_user(test_db):
    """Cria user de teste pre-registado"""
    user = User(email="test@example.com", password_hash="hashed_password")
    test_db.add(user)
    test_db.commit()
    return user

@pytest.fixture
def test_process(test_db, test_user):
    """Cria process de teste com status=CREATED"""
    process = Process(
        title="Test JD",
        jd_text="Python, SQL, FastAPI required",
        owner_id=test_user.id,
        status=ProcessStatus.CREATED
    )
    test_db.add(process)
    test_db.commit()
    return process

@pytest.fixture
def test_candidates(test_db, test_process):
    """Cria 5 candidates de teste"""
    candidates = []
    for i in range(5):
        candidate = Candidate(
            name=f"Test Candidate {i}",
            process_id=test_process.id,
            file_path=f"/tmp/test_{i}.pdf"
        )
        test_db.add(candidate)
        candidates.append(candidate)
    test_db.commit()
    return candidates

@pytest.fixture
def jwt_token(test_user):
    """Gera JWT válido para test_user"""
    from backend.api.services.auth_service import create_access_token
    return create_access_token(user_id=test_user.id)

@pytest.fixture
def auth_header(jwt_token):
    """Authorization header com JWT"""
    return {"Authorization": f"Bearer {jwt_token}"}
```

**Nota importante (para evitar confusão):**
- O fixture `client` é um `TestClient`, usado para executar rotas async de forma síncrona.
- Mas `response = client.get(...)` / `client.post(...)` devolve um **`httpx.Response`**.
- Portanto, os helpers de validação em `backend/api/tests/helpers/assertions.py` devem receber/tipar `response` como `httpx.Response` (não `TestClient`).

---

## 9. Execução de Testes

### Comando para rodar todos os testes:

```bash
# Unit tests
pytest backend/api/tests/unit/ -v --cov=backend/api --cov-report=html

# Integration tests
pytest backend/api/tests/integration/ -v --tb=short

# Security tests
pytest backend/api/tests/security/ -v

# Load tests (opcional, mais lento)
pytest backend/api/tests/load/ -v -m "load" --durations=10

# Tudo junto (com coverage report)
pytest backend/api/tests/ -v --cov=backend/api --cov-report=html --cov-fail-under=70
```

### Alvo de Cobertura

| Componente | Cobertura Mínima | Prioridade |
|---|---|---|
| Services | 85% | ALTA |
| Routes | 80% | ALTA |
| Utils | 90% | ALTA |
| DB Models | 70% | MÉDIA |
| Overall | 70% | ALTA |

---

## 10. Checklist de Testes por Sprint

| Sprint | Componente | Unit Tests | Integration | Security | Status |
|---|---|---|---|---|---|
| 0 | Infraestrutura | N/A | ✓ | N/A | — |
| 1 | Database | N/A | ✓ | ✓ | — |
| 2 | Schemas/Utils | ✓ | N/A | N/A | — |
| 3 | Process/Upload | ✓ | ✓ | ✓ | — |
| 4 | Auth | ✓ | ✓ | ✓ | — |
| 5 | Screening | ✓ | ✓ | N/A | — |
| 6 | Reports | ✓ | ✓ | N/A | — |
| 7 | All | ✓ | ✓ | ✓ | IN PROGRESS |

---

## 11. CI/CD Integration

Adicionar a `.github/workflows/ci.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test

    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          pytest backend/api/tests/ -v --cov=backend/api --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

## Resumo

- **Total de testes recomendados:** ~80-100 testes
- **Tempo de execução:** ~5-10 minutos (unit + integration)
- **Cobertura alvo:** 70%+ (all components)
- **Security:** 5 categorias (JWT, file upload, path traversal, SQL injection, XSS)
- **Performance:** 4 testes load (CSV export, concurrent, file upload, DB performance)

**Sprint 7 pronto para ser implementado! 🎯**
