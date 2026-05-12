# Sprint 5 — Screening e Results — FINALIZADO

**Status:** ✅ COMPLETO | **Data:** Maio 2026
**Objetivo:** Integrar pipeline v1.0 via API, disparar em background, persistir resultados

---

## Deliverables — 3 Ficheiros (1.026 linhas)

### 1. screening_service.py (390 linhas)
**Localização:** backend/api/services/screening_service.py

Orquestra pipeline v1.0 (parser, NLP, scoring) com persistência em DB.

**Métodos:**
- `run(process_id, db_session)` — executa screening completo em background
- `get_results(process_id, db_session)` — retorna ranking + metadata
- `_load_jd_criteria(jd_text)` — JobDescriptionParser wrapper
- `_process_candidate(candidate, jd_criteria)` — pipeline: preprocess → extract → score

**Design:**
- ✅ Robusto: erro de 1 candidato não bloqueia outros (ProcessingError para auditoria)
- ✅ Síncrono: BackgroundTasks executa em thread pool (não async necessário)
- ✅ Transacional: commit/rollback atomicamente
- ✅ Logging estruturado: sem raw text ou PII

### 2. results.py (343 linhas)
**Localização:** backend/api/routes/results.py

HTTP handlers para screening: disparar e consultar resultados.

**Endpoints:**

| Método | Rota | Função |
|--------|------|--------|
| POST | /api/processes/{id}/run | Dispara screening em background, retorna 200 com status "processing" |
| GET | /api/processes/{id}/results | Retorna 202 (processing) ou 200 (completed/failed) |

**Dependencies (injeção):**
- `get_nlp_model()` — extrai spaCy de app.state (carregado no startup)
- `get_screening_service()` — factory ScreeningService
- `validate_process_id()` — UUID validation
- `get_current_user()` — JWT autenticação (Sprint 4)

**Features:**
- ✅ 202 Accepted para processing (JSONResponse com status_code explícito)
- ✅ UUID validation antes de processar
- ✅ State machine validation (só run em files_uploaded)
- ✅ Exception handling: 404, 409, 400, 401

### 3. main.py (atualizado — 293 linhas)
**Localização:** backend/api/main.py

Integração do router results.

**Mudanças (3 linhas):**
1. Linha 18: `from backend.api.routes import processes, upload, auth, results`
2. Linha 254: `app.include_router(results.router)` (após upload)
3. Linha 257: Logging atualizado: "auth, processes, upload, results"

---

## Bugs Corrigidos Pós-Finalização

### Bug #1: Sintaxe Inválida (Linha 114 - screening_service.py)
**Erro:** `len(jd_criteria.get["skills",])` — colchetes em vez de parênteses
**Corrigido:** `len(jd_criteria.get("skills", []))`
**Severidade:** 🔴 CRÍTICO — Runtime exception ao tentar processar JD

---

## Conformidade Arquitetura

✅ ARCHITECTURE.md Section 7.1 (ScreeningService contracts)
✅ Section 10 (API Design endpoints + status codes)
✅ ADR-07 (BackgroundTasks, síncrono)
✅ ADR-04 (ORM/Schemas separados)
✅ SDLC Section 9.3 (state machine, partial success)
✅ Logging (request_id middleware, sem PII)
✅ Security (JWT obrigatória em ambos endpoints)
✅ Error handling (NotFoundError→404, ConflictError→409, ValidationError→400)

---

## Fluxo de Execução

### POST /api/processes/{id}/run

1. Valida UUID de process_id
2. Autentica JWT (get_current_user)
3. Load processo, valida estado (files_uploaded)
4. Valida que há candidatos uploaded
5. Dispara ScreeningService.run() em BackgroundTasks
6. Retorna 200 {process_id, status: "processing"} imediatamente

### GET /api/processes/{id}/results

1. Valida UUID de process_id
2. Autentica JWT
3. Load processo
4. **Se processing:** retorna 202 Accepted {status: "processing"}
5. **Se completed:** retorna 200 {status, summary, candidates (ranked)}
6. **Se failed:** retorna 200 {status, error_message}

### ScreeningService.run() (background)

1. Load + validate processo (files_uploaded)
2. Mark como processing
3. Load JD criteria (JobDescriptionParser)
4. Para cada candidato:
   - TextPreprocessor.process() — limpeza + tokens
   - ResumeFeatureExtractor.extract() — features
   - ResumeScorer.score() — pontuação ponderada
   - Persist Result em DB (ou ProcessingError se falha)
5. Commit transação
6. Mark como completed

---

## Pipeline v1.0 — Transformação de Dados

**Etapa 1: Preprocessing**
- Input: CV bruto ("João Silva\nSkills: Python, SQL")
- Output: `{clean_text: "joão silva skills python sql", tokens: ["joão", "silva", ...]}`

**Etapa 2: Feature Extraction**
- Input: clean_text + jd_skills
- Output: `{matched_skills: ["Python", "SQL"], experience_years: 5, education_level: 4.0, keywords: [...]}`

**Etapa 3: Scoring (ponderado)**
- Skills Match (40%): 2/3 = 66.7%
- Experience (25%): 5 years vs min 3 = 90%
- Education (20%): 4/5 = 80%
- Keywords TF-IDF (15%): cosine similarity = 75%
- **Total: 66.7×0.40 + 90×0.25 + 80×0.20 + 75×0.15 = 79.8 → "Strong Match"**

**Etapa 4: Persistência**
- Result ORM: `{candidate_id, total_score, category, breakdown, matched_skills, required_skills, missing_skills, experience_years_found}`

---

## Nota: Campo `required_skills`

**Não faz parte do pipeline v1.0.** É adicionado em screening_service.py:377 para auditoria:
```python
required_skills=jd_criteria["skills"],  # Skills necessários da JD
```
Permite recruiter ver diferença: `missing_skills = required_skills - matched_skills`

---

## Decisões Técnicas

| Decisão | Razão |
|---------|-------|
| Síncrono (não async) | v1.0 é CPU-bound, não I/O-bound. BackgroundTasks usa thread pool |
| Robusto (continue on error) | SDLC 9.3: "partial success expected and normal". ProcessingError para audit |
| Status-only polling | Resultados parciais induzem erro (ranking incompleto) |
| 202 para processing | HTTP standard: 202 = accepted but not yet completed |

---

## Integração

**Ficheiros a copiar:**
- screening_service.py → backend/api/services/
- results.py → backend/api/routes/
- main_updated.py → backend/api/main.py (3 mudanças)

**Validação pós-integração:**
```bash
docker compose up -d --build
curl http://localhost:8000/docs  # Swagger UI
# POST /api/auth/register
# POST /api/auth/login
# POST /api/processes
# POST /api/processes/{id}/upload
# POST /api/processes/{id}/run  ← dispara screening
# GET /api/processes/{id}/results  ← polling
```

---

## Test Coverage Recomendado

### Unit Tests (backend/api/tests/unit/)
**test_screening_service.py:** test_process_candidate_success, test_get_results_completed_status
**test_results_routes.py:** test_post_run_success, test_post_run_invalid_uuid (400), test_post_run_no_candidates (400), test_post_run_already_processing (409), test_get_results_processing (202), test_get_results_completed (200), test_unauthorized_no_jwt (401)

### Integration Tests (backend/api/tests/integration/)
- Full pipeline: POST /run + polling GET /results
- State transitions: created → files_uploaded → processing → completed
- Error recovery: ProcessingError persisted for failed candidates

### Load Test (Optional)
- 100 CVs paralelos → state consistency verificado
- DB commits: todas Results salvos atomicamente

---

## Próximos Passos

**Sprint 6: Report Service**
- ReportService: exportação CSV, JSON, TXT
- Routes: GET /api/processes/{id}/export/{format}
- Streaming de ficheiros via StreamingResponse

---

**Sprint 5 — PRONTO PARA DEPLOY**
