# Sprint 6 — Export Flowchart (Diagramático)

## 1. Client Request

```
┌─────────────────────────────────────────────────────────────────────────┐
│ CLIENT REQUEST                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GET /api/processes/{process_id}/export/csv                           │
│  Headers: Authorization: Bearer <JWT_TOKEN>                           │
│                                                                         │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │
                              ▼
```

---

## 2. FastAPI Dependency Injection

```
┌─────────────────────────────────────────────────────────────────────────┐
│ DEPENDENCY INJECTION LAYER                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ① validate_process_id(process_id)                                    │
│     └─ UUID.parse() check → ValidationError (400) if invalid          │
│                                                                         │
│  ② get_current_user(JWT)                                              │
│     └─ JWT validation → UnauthorizedError (401) if invalid            │
│                                                                         │
│  ③ get_db()                                                            │
│     └─ SQLAlchemy Session → HTTPException (500) if DB unavailable    │
│                                                                         │
│  ④ get_report_service()                                               │
│     └─ ReportService() instantiation (stateless factory)              │
│                                                                         │
│  Result: All dependencies validated ✅                                 │
│                                                                         │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │
                              ▼
```

---

## 3. Route Handler: export_csv()

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ROUTE HANDLER: export_csv()                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  @router.get("/{process_id}/export/csv")                              │
│  async def export_csv(                                                 │
│      process_id: str,              (from path, already UUID-validated) │
│      current_user: User,           (from JWT, already authenticated)   │
│      db_session: Session,          (from get_db)                      │
│      report_service: ReportService (from factory)                     │
│  ) -> StreamingResponse:                                               │
│                                                                         │
│  try:                                                                  │
│      file_bytes, filename = report_service.export_csv(                │
│          process_id,                                                  │
│          db_session                                                   │
│      )                                                                 │
│                                                                         │
│  Result: Tuple[bytes, str] returned                                   │
│                                                                         │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │
                              ▼
```

---

## 4. ReportService.export_csv() — Phase 1: Validation

```
┌─────────────────────────────────────────────────────────────────────────┐
│ REPORT_SERVICE.EXPORT_CSV() — PHASE 1: VALIDATION                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  _load_and_validate_process(process_id, db_session)                   │
│  │                                                                     │
│  ├─ Step 1: Query processo                                            │
│  │  Query: SELECT * FROM processes WHERE id = $1                      │
│  │  Result: Process ORM object ← or None                              │
│  │                                                                     │
│  └─ Step 2a: Validar existe                                           │
│     if not process:                                                   │
│         └─ raise NotFoundError(404)                                   │
│            "Process with ID ... not found"                            │
│                                                                         │
│  └─ Step 2b: Validar status == COMPLETED                              │
│     if process.status != ProcessStatus.COMPLETED:                     │
│         └─ raise ValidationError(400)                                 │
│            "Cannot export from process in '{status}' state.           │
│             Process must be completed."                               │
│                                                                         │
│  Result: Process ORM object (validated) ✅                            │
│                                                                         │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │
                              ▼
```

---

## 5. ReportService.export_csv() — Phase 2: Build Candidates List

```
┌─────────────────────────────────────────────────────────────────────────┐
│ REPORT_SERVICE.EXPORT_CSV() — PHASE 2: BUILD CANDIDATES                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  _build_candidates_list(process: Process, db_session)                 │
│  │                                                                     │
│  ├─ Step 1: Query Results (ranked)                                    │
│  │  SELECT r.id, r.total_score, r.category, r.breakdown,             │
│  │         r.matched_skills, r.missing_skills,                       │
│  │         r.experience_years_found                                   │
│  │  FROM results r                                                    │
│  │  WHERE r.candidate_id IN (                                         │
│  │      SELECT id FROM candidates WHERE process_id = $1               │
│  │  )                                                                  │
│  │  ORDER BY r.total_score DESC                                       │
│  │  Result: list[Result] (ordered by score)                          │
│  │                                                                     │
│  ├─ Step 2: Loop & Build dicts                                        │
│  │  for result in results:                                            │
│  │      candidate = Query Candidate by result.candidate_id           │
│  │      append({                                                      │
│  │          "name": candidate.name,                                  │
│  │          "total_score": result.total_score,                       │
│  │          "category": result.category,                             │
│  │          "breakdown": result.breakdown,                           │
│  │          "matched_skills": result.matched_skills,                 │
│  │          "missing_skills": result.missing_skills,                 │
│  │          "experience_years_found": result.experience_years_found  │
│  │      })                                                            │
│  │                                                                     │
│  Result: list[dict] (ReportGenerator format) ✅                       │
│                                                                         │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │
                              ▼
```

---

## 6. ReportService.export_csv() — Phase 3: Generate & Stream

```
┌─────────────────────────────────────────────────────────────────────────┐
│ REPORT_SERVICE.EXPORT_CSV() — PHASE 3: GENERATE & STREAM                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  with tempfile.TemporaryDirectory() as temp_dir:                      │
│  │   # temp_dir = "/tmp/tmpXYZ123abc/" (auto-cleanup on exit)        │
│  │                                                                     │
│  ├─ Step 1: Instantiate ReportGenerator                               │
│  │  generator = ReportGenerator(temp_dir)                             │
│  │  Result: Generator ready to format                                 │
│  │                                                                     │
│  ├─ Step 2: Generate CSV file                                         │
│  │  path = generator.save_csv(candidates)                             │
│  │  # Writes: /tmp/tmpXYZ123abc/results_20260505_143022.csv          │
│  │  Result: path (Path or string)                                     │
│  │                                                                     │
│  ├─ Step 3: Read file as bytes                                        │
│  │  with open(path, 'rb') as f:                                       │
│  │      file_bytes = f.read()                                         │
│  │  Result: file_bytes (b'Rank,Candidate,...')                       │
│  │                                                                     │
│  └─ Step 4: Exit context (auto-cleanup)                               │
│     # /tmp/tmpXYZ123abc/ and all contents DELETED ✨                  │
│                                                                         │
│  Return: (file_bytes, filename)                                        │
│                                                                         │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │
                              ▼
```

---

## 7. Route Handler — HTTP Response

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ROUTE HANDLER: HTTP RESPONSE CONSTRUCTION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  return StreamingResponse(                                             │
│      iter([file_bytes]),                                              │
│      #   ↑ Iterável com bytes                                         │
│      #   FastAPI consome em chunks (streaming)                        │
│                                                                         │
│      media_type="text/csv; charset=utf-8",                            │
│      #   ↑ MIME type: diz ao browser que é CSV                        │
│      #   charset=utf-8 para suportar acentos                          │
│                                                                         │
│      headers={                                                         │
│          "Content-Disposition":                                       │
│              f'attachment; filename="{filename}"'                     │
│          #   ↑ attachment = força download                            │
│          #   filename = nome sugerido para o ficheiro                 │
│      },                                                                │
│  )                                                                     │
│                                                                         │
│  Result: StreamingResponse object                                      │
│                                                                         │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │
                              ▼
```

---

## 8. HTTP Response (Wired to Client)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HTTP RESPONSE (200 OK)                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ HTTP/1.1 200 OK                                                        │
│ Content-Type: text/csv; charset=utf-8                                  │
│ Content-Disposition: attachment; filename="results_20260505_143022.csv"│
│ Content-Length: 2048                                                   │
│ Transfer-Encoding: chunked                                             │
│                                                                         │
│ [streaming CSV bytes]                                                  │
│ Rank,Candidate,Total Score,Category,Matched Skills,...                │
│ 1,João Silva,85.5,Strong Match,"Python, SQL, FastAPI",...            │
│ 2,Maria Santos,72.3,Potential Match,"Python, SQL",...                 │
│ ...                                                                     │
│                                                                         │
└─────────────────────────────┬─────────────────────────────────────────┘
                              │
                              ▼
```

---

## 9. Client Receives Download

```
┌─────────────────────────────────────────────────────────────────────────┐
│ CLIENT RECEIVES & DOWNLOADS                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Browser (Chrome, Firefox, Safari, etc)                                │
│  │                                                                     │
│  ├─ Parses response headers                                            │
│  │  Content-Disposition: attachment → DOWNLOAD (não open)            │
│  │  filename: results_20260505_143022.csv → use como nome             │
│  │                                                                     │
│  ├─ Streams bytes ao disco                                             │
│  │  ~/Downloads/results_20260505_143022.csv (no buffer)               │
│  │                                                                     │
│  └─ User action: ficheiro pronto para usar                             │
│     Abrir em Excel, Google Sheets, ou text editor                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Error Paths (Early Returns)

### Path 1: Process não existe

```
GET /api/processes/INVALID-UUID/export/csv
                    ↓
validate_process_id()
    └─ UUID.parse() raises ValueError
    └─ Catch + raise ValidationError
                    ↓
Exception handler (400 Bad Request)
                    ↓
Response:
{
  "detail": "Invalid process_id format: INVALID-UUID. Must be a valid UUID."
}
```

### Path 2: Process não existe em DB

```
_load_and_validate_process(process_id)
    └─ Query: SELECT * FROM processes WHERE id = $1
    └─ Result: None (not found)
    └─ raise NotFoundError
                    ↓
Exception handler (404 Not Found)
                    ↓
Response:
{
  "detail": "Process with ID 550e8400-e29b-41d4-a716-446655440000 not found."
}
```

### Path 3: Process não em estado COMPLETED

```
_load_and_validate_process(process_id)
    └─ Query OK, process exists ✅
    └─ Check: process.status == ProcessStatus.COMPLETED
    └─ Status is "processing" ❌
    └─ raise ValidationError
                    ↓
Exception handler (400 Bad Request)
                    ↓
Response:
{
  "detail": "Cannot export from process in 'processing' state. Process must be completed."
}
```

### Path 4: JWT inválido

```
GET /api/processes/{id}/export/csv
(sem Authorization header, ou token expirado)
                    ↓
get_current_user() dependency
    └─ JWT validation fails
    └─ raise UnauthorizedError
                    ↓
Exception handler (401 Unauthorized)
                    ↓
Response:
{
  "detail": "Not authenticated"
}
```

### Path 5: Erro inesperado (Database connection lost)

```
_build_candidates_list()
    └─ Query: SELECT * FROM results WHERE ...
    └─ Database connection lost ❌
    └─ SQLAlchemy raises DatabaseError
                    ↓
except Exception as e:
    └─ logger.error(...)
    └─ raise HTTPException(status_code=500)
                    ↓
Exception handler (500 Internal Server Error)
                    ↓
Response:
{
  "detail": "An unexpected error occurred while exporting CSV."
}
```

---

## Database Queries (SQL Pseudocode)

### Query 1: Load & Validate Process

```sql
-- _load_and_validate_process()
SELECT id, status, title, created_at, updated_at, jd_text
FROM processes
WHERE id = $1  -- process_id
LIMIT 1;
```

### Query 2: Build Candidates List

```sql
-- _build_candidates_list()
SELECT 
    r.id,
    r.total_score,
    r.category,
    r.breakdown,
    r.matched_skills,
    r.missing_skills,
    r.experience_years_found,
    c.id as candidate_id,
    c.name
FROM results r
INNER JOIN candidates c ON r.candidate_id = c.id
WHERE c.process_id = $1  -- process_id
ORDER BY r.total_score DESC;
```

**Indexes needed:**
```
- PRIMARY KEY (processes.id)
- PRIMARY KEY (candidates.id)
- PRIMARY KEY (results.id)
- FOREIGN KEY (candidates.process_id) → processes.id
- FOREIGN KEY (results.candidate_id) → candidates.id
- INDEX (results.total_score DESC)  -- for ORDER BY
```

---

## Summary: All Paths

```
Request
  │
  ├─ Path 1: UUID inválido → 400 ValidationError
  ├─ Path 2: JWT inválido → 401 UnauthorizedError
  ├─ Path 3: Process não existe → 404 NotFoundError
  ├─ Path 4: Process não completed → 400 ValidationError
  ├─ Path 5: DB error → 500 HTTPException
  │
  └─ Path 6: Success → 200 OK + StreamingResponse
             → CSV/JSON/TXT ficheiro para download
```

---

**Sprint 6 — Fluxo de Export Documentado ✅**
