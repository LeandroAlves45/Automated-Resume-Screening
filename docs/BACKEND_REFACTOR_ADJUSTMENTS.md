# Ajustes do backend — refatoração pós-revisão

Este documento resume as alterações aplicadas ao backend em **cinco** rondas de revisão: (1) bugs críticos, segurança e DRY; (2) wiring da aplicação, propagação de erros HTTP e enum `cancelled`; (3) máquina de estados do screening, contrato de erros na BD e pequenos ajustes de testes/rotas; (4) concorrência/idempotência no runner, sanitização da mensagem fatal, semântica HTTP correta em erros de BD e testes de regressão; (5) recuperação de processos órfãos, normalização de exceções não-API, pequenas correções de semântica HTTP e smoke tests de autenticação. Serve como base para atualizar README, guias de deploy ou documentação técnica.

---

## 1. Health check e base de dados

| Antes | Depois |
|-------|--------|
| `check_db_connection()` devolvia `dict` (`{"connected": ...}`), sempre truthy em `if not check_db_connection()` | Devolve `bool` (`True` / `False`) |

**Ficheiros:** `backend/api/db/database.py`, `backend/api/main.py`

- Startup e `GET /health` passam a refletir corretamente falhas de ligação à BD.

---

## 2. Handlers de exceções HTTP

| Antes | Depois |
|-------|--------|
| Vários handlers; `UnauthorizedError` sem `JSONResponse`; handler genérico usava `exc.HTTP_401_UNAUTHORIZED` (inexistente) → `AttributeError` e 500 em casos como `ForbiddenError` | Um único `api_exception_handler` para `BaseAPIException` com `status_code` e `error_code` lidos da exceção |

**Ficheiro:** `backend/api/main.py`

- `ForbiddenError` (403), `UnauthorizedError` (401), `ValidationError` (400), etc. respondem com JSON coerente sem 500 indevidos.

---

## 3. Categorias de match (scorer vs resumo)

| Antes | Depois |
|-------|--------|
| `screening_service` comparava strings erradas (`"Strong match"`, …) vs scorer (`"Strong Match"`, …) → contagens no `summary` sempre zero | Constantes partilhadas `MATCH_CATEGORIES` em `backend/api/scoring_config.py`; scorer, agregação de resultados e relatórios usam a mesma fonte |

**Ficheiros:** `backend/api/scoring_config.py`, `backend/scoring/scorer.py`, `backend/api/services/screening_service.py`, `backend/reports/reporter.py`

**Teste:** `backend/api/tests/unit/services/test_screening_service.py` — `test_get_results_summary_counts_categories`

---

## 4. `missing_skills` nos resultados

| Antes | Depois |
|-------|--------|
| Código acedia a `result.missing_skills` no ORM `Result`, mas a coluna não existe | `compute_missing_skills(result)` deriva `required_skills - matched_skills` |

**Ficheiros:** `backend/api/services/results_query.py`, `backend/api/services/screening_service.py`, `backend/api/services/report_service.py`

---

## 5. Configuração e arranque

| Tema | Alteração |
|------|-----------|
| `create_tables()` no lifespan | Em produção (`app_env == production`) **não** corre; schema deve ser gerido com Alembic |
| `database_url` / `jwt_secret_key` em produção | Validadores rejeitam placeholders de desenvolvimento explícitos; JWT em produção continua a exigir ≥ 32 caracteres |

**Ficheiros:** `backend/api/main.py`, `backend/api/config.py`

- Constantes de placeholder: `DEV_DATABASE_URL_PLACEHOLDER`, `DEV_JWT_SECRET_PLACEHOLDER` em `config.py`.

---

## 6. Consistência de timestamps

- `Result` criado no pipeline: `created_at` passou de `datetime.utcnow()` para `datetime.now(timezone.utc)`.

**Ficheiro:** `backend/api/services/screening_service.py`

---

## 7. Rotas e dependências (DRY)

| Novo / alterado | Descrição |
|-----------------|-----------|
| `backend/api/routes/deps.py` | Factories partilhadas: `get_process_service`, `get_candidate_service`, `get_nlp_model`, `get_screening_service`, `get_report_service`, `validate_process_id` |
| `backend/api/routes/processes.py` | Usa `deps`; `ProcessResponse.from_orm_process(process)` em vez de mapeamento repetido |
| `backend/api/routes/upload.py` | Usa `deps.get_candidate_service` |
| `backend/api/routes/results.py` | Usa `deps`; helper `_stream_export` para CSV/JSON/TXT; `BackgroundTasks` sem default `None` |

---

## 8. Schema Pydantic — processos

- `ProcessResponse.from_orm_process(cls, process)` em `backend/api/models/schemas.py` para mapeamento ORM → API centralizado.

---

## 9. Query partilhada de resultados

| Novo | Descrição |
|------|-----------|
| `backend/api/services/results_query.py` | `fetch_ranked_results(process_id, db_session)` — JOIN `Result` + `Candidate`, ordenado por `total_score` DESC (evita N+1 e duplicação entre screening e relatórios) |

Consumidores: `ScreeningService.get_results`, `ReportService._build_candidates_list`.

---

## 10. Verificação (1.ª ronda)

- Suite: `pytest backend` — todos os testes de serviço a passar (ambiente com `venv` e dependências instaladas).

---

---

# Segunda ronda — Revisão adicional do backend

Alterações aplicadas após uma segunda passagem de revisão ao código. Cada item corresponde a um ponto identificado no plano *"Revisão backend adicional"*.

---

## 11. Router de resultados não montado (crítico)

| Antes | Depois |
|-------|--------|
| `main.py` incluía `auth`, `processes` e `upload`, mas **nunca** `results.router`; `POST /run`, `GET /results`, `GET /export/*` respondiam 404 em produção | `app.include_router(results.router)` adicionado; log de startup atualizado para `auth, processes, upload, results` |

**Ficheiro:** `backend/api/main.py`

---

## 12. `ForbiddenError` convertido em HTTP 400 em `processes.py` (alto)

| Antes | Depois |
|-------|--------|
| `get_process`, `create_process`, `list_processes`: bloco `except Exception → ValidationError` engolia `ForbiddenError` e devolvia **400** em vez de **403** | `except BaseAPIException: raise` antes do catch genérico; catch genérico agora levanta `InternalServerError` (500) |

**Ficheiro:** `backend/api/routes/processes.py`

---

## 13. `ForbiddenError` convertido em HTTP 500 em `results.py` (alto)

| Antes | Depois |
|-------|--------|
| `run_screening` e `get_results` só re-lançavam `NotFoundError`/`ConflictError`/`ValidationError`; `ForbiddenError` caía no `except Exception → HTTPException(500)` | `except BaseAPIException: raise` em todos os blocos; `HTTPException` substituído por `InternalServerError`; contrato `{ detail, error_code }` uniforme em toda a API |

**Ficheiro:** `backend/api/routes/results.py`

---

## 14. Inconsistência `ProcessStatus.CANCELLED` entre ORM e schema (alto)

| Camada | Antes | Depois |
|--------|-------|--------|
| `backend/api/db/models.py` | `"cancelled"` | `"cancelled"` (inalterado — fonte de verdade) |
| `backend/api/models/schemas.py` | `"canceled"` (typo, falta um `l`) | `"cancelled"` |

Obter ou listar um processo cancelado gerava falha de validação Pydantic na serialização da resposta. Alinhar os dois enums resolve o problema sem alterar a base de dados.

**Ficheiro:** `backend/api/models/schemas.py`

---

## 15. `IndentationError` latente em `errors.py`

| Antes | Depois |
|-------|--------|
| Docstring de `BaseAPIException` com indentação de 4 espaços; métodos da classe a 2 espaços → `IndentationError` ao importar o módulo (manifesta-se quando `results` é incluído na cadeia de importação) | Docstring alinhada a 2 espaços, consistente com todas as subclasses do ficheiro |

**Ficheiro:** `backend/api/utils/errors.py`

---

## 16. Testes HTTP de rotas (novos)

Criados testes de integração HTTP onde antes só existiam stubs vazios.

| Ficheiro | O que cobre |
|----------|-------------|
| `backend/api/tests/integration/routes/conftest.py` | Override de `get_nlp_model` para injetar o modelo spaCy real nas rotas, sem depender do lifespan |
| `backend/api/tests/integration/routes/test_process_routes.py` | 200 para owner; **403** para utilizador diferente (`ForbiddenError` não deve tornar-se 400); 404 para UUID inexistente; serialização de processo `cancelled` sem erro Pydantic |
| `backend/api/tests/integration/routes/test_upload_results_routes.py` | Smoke em `POST /run` e `GET /results`: 404, **403** e 409 chegam ao cliente com `error_code` correto |

*(Nota: os testes HTTP de rotas vivem em `integration/routes/`, não em `unit/routes/`.)*

---

## 17. Verificação (2.ª ronda)

- Suite: `pytest backend/api/tests/` — todos os testes a passar após esta ronda (ver secção 20 para contagem atual).

---

# Terceira ronda — Screening, rotas e testes (pós-revisão)

Correções alinhadas com a revisão de código que identificou `mark_failed` com estado inválido, `completed` sem resultados, filtro por `process_id` na rota, mensagens `ProcessingError` com possível vazamento de dados, e legibilidade do `conftest`.

---

## 18. `ScreeningService.run` — máquina de estados e resultados

| Tema | Antes | Depois |
|------|-------|--------|
| Processo ≠ `files_uploaded` dentro do runner | Chamava `mark_failed`, mas só é válido a partir de `processing` → `ConflictError` ou estado inconsistente | **Log + return** sem alterar o processo (defesa contra corrida / chamada fora da rota) |
| Lista de candidatos vazia após `mark_processing` | `mark_completed` | `mark_failed` com mensagem explícita |
| Todos os candidatos falham no loop | `mark_completed` sem nenhum `Result` | `mark_failed` com mensagem que remete a `processing_errors` |
| `ProcessingError.message` | Incluía `str(e)` (risco de excertos de CV / PII na BD) | Mensagem genérica: `{TypeName}: candidate processing failed`; detalhe continua só nos **logs** |
| ORM `Result` | Construtor passava `missing_skills=` (coluna inexistente no modelo) | Argumento removido; `missing_skills` continua derivado em API via `compute_missing_skills` |
| Exceções de domínio no `run()` | Só `NotFoundError` e `ForbiddenError` re-lançadas | **`BaseAPIException`** re-lançada (incl. `ConflictError`, etc.) |

**Ficheiro:** `backend/api/services/screening_service.py`

---

## 19. `POST /run` — filtro de candidatos

| Antes | Depois |
|-------|--------|
| `Candidate.process_id == process_id` (string do path) | `Candidate.process_id == process.id` (UUID do ORM, consistente com o serviço) |

**Ficheiro:** `backend/api/routes/results.py`

---

## 20. Testes e fixture HTTP

| Alteração | Detalhe |
|-----------|---------|
| `test_screening_service.py` | `test_run_skips_when_process_not_files_uploaded`, `test_run_all_candidates_fail_marks_process_failed` |
| `conftest.py` (`test_process`) | `jd_text` como **uma** string entre parênteses (legibilidade; comportamento idêntico) |

**Ficheiros:** `backend/api/tests/unit/services/test_screening_service.py`, `backend/api/tests/conftest.py`

---

## 21. Verificação (3.ª ronda)

- Suite: `pytest backend/api/tests/` — **70 / 70 testes a passar**.

---

# Quarta ronda — Concorrência, sanitização e semântica HTTP de BD

Correções identificadas na revisão *"Revisão backend N+1"*: idempotência no runner para estado `processing`, sanitização da mensagem de erro fatal, mapeamento correto de `SQLAlchemyError` para HTTP 500 e testes de regressão dirigidos.

---

## 22. `ScreeningService.run` — idempotência para estado `processing` e `ConflictError` para estados inesperados

| Estado ao entrar no `run()` | Antes | Depois |
|------------------------------|-------|--------|
| `PROCESSING` (corrida de dois workers) | **log + return silencioso** — nenhuma sinalização ao chamador | **log + return silencioso** — comportamento idêntico mas documentado como idempotência intencional |
| Outro estado inesperado (`CREATED`, `COMPLETED`, `FAILED`, `CANCELLED`) | **log + return silencioso** — chamador não sabia que o processamento não ocorreu | **`ConflictError` (409)** com mensagem descritiva do estado atual |

O estado `PROCESSING` mantém o retorno silencioso para permitir que um segundo worker se retire sem sobrescrever o estado do primeiro. Qualquer outro estado inesperado passa a lançar `ConflictError`, tornando o erro visível para o chamador.

**Ficheiro:** `backend/api/services/screening_service.py`

---

## 23. Mensagem fatal de `mark_failed` sanitizada

| Antes | Depois |
|-------|--------|
| `error_message=f"Screening failed: {str(e)}"` — conteúdo da exceção exposto no ORM e potencialmente na API via `GET /results` | Mensagem genérica `"Screening failed due to an unexpected internal error."` persistida no processo; `logger.exception(...)` regista o detalhe completo (incluindo traceback) nos logs |

Alinha o ramo fatal com a política já aplicada ao loop de candidatos (`ProcessingError` com mensagem genérica).

**Ficheiro:** `backend/api/services/screening_service.py`

---

## 24. `SQLAlchemyError` → `InternalServerError` (HTTP 500) nos serviços

| Serviço / método | Antes | Depois |
|------------------|-------|--------|
| `ProcessService.create_process` | `ValidationError` (400) | `InternalServerError` (500) |
| `ProcessService.list_processes` | `ValidationError` (400) | `InternalServerError` (500) |
| `ProcessService.get_process` | `ValidationError` (400) | `InternalServerError` (500) |
| `ProcessService.update_status` | `ValidationError` (400) | `InternalServerError` (500) |
| `CandidateService.list_candidates` | `ValidationError` (400) | `InternalServerError` (500) |
| `CandidateService.get_candidate` | `ValidationError` (400) | `InternalServerError` (500) |

Falhas de infraestrutura (BD indisponível, timeout, constraint violation) não são erros de validação de input. HTTP 400 é incorreto semanticamente; HTTP 500 sinaliza corretamente ao cliente que houve uma falha interna.

**Ficheiros:** `backend/api/services/process_service.py`, `backend/api/services/candidate_service.py`

---

## 25. Testes de regressão (4.ª ronda)

| Teste (novo / atualizado) | O que verifica |
|---------------------------|----------------|
| `test_run_skips_silently_when_already_processing` (substituiu `test_run_skips_when_process_not_files_uploaded`) | Runner em estado `PROCESSING` retorna silenciosamente; estado não é alterado |
| `test_run_raises_conflict_for_unexpected_state` (novo) | Estado `CREATED` (e análogos) levanta `ConflictError`; processo permanece inalterado |
| `test_run_fatal_error_message_is_sanitized` (novo) | `process.error_message` após erro fatal **não** contém o detalhe técnico da exceção |

**Ficheiro:** `backend/api/tests/unit/services/test_screening_service.py`

---

## 26. Verificação (4.ª ronda)

- Suite: `pytest backend/api/tests/` — **72 / 72 testes a passar**.

---

## Quinta ronda — N+2

### 27. Reconciliação de processos órfãos em `processing`

| Antes | Depois |
|-------|--------|
| Processos que ficavam em `processing` (worker morreu após `mark_processing`) nunca eram recuperados | `ProcessService.reconcile_stuck_processes(timeout_minutes)` marca como `failed` todos os processos em `processing` há mais do que N minutos |

**Ficheiros:** `backend/api/services/process_service.py`, `backend/api/main.py`, `backend/api/config.py`

- Nova opção `STUCK_PROCESS_TIMEOUT_MINUTES` (default: 30) em `Settings`.
- Chamada automática no startup da aplicação; falhas de reconciliação são logadas mas não bloqueiam o arranque.
- A `error_message` gerada explica ao cliente que o screening expirou e sugere retentar.

---

### 28. Normalização de exceções fora de `BaseAPIException`

| Local | Antes | Depois |
|-------|-------|--------|
| `routes/deps.py` — `get_nlp_model` | `RuntimeError` se modelo ausente → 500 sem contrato JSON | `SpacyModelError` (subclasse de `BaseAPIException`) → resposta JSON uniforme |
| `services/auth_service.py` — `AuthService.register` | `db.commit()` sem `try/except` → `SQLAlchemyError` podia propagar sem mapping | Commit envolvido em `try/except`; mapeia para `InternalServerError` com log e rollback |
| `services/candidate_service.py` — `save_file` | `RuntimeError` em erros de I/O e BD | `InternalServerError` em todos os caminhos de erro de disco/BD |
| `routes/upload.py` | `except Exception` engolia `RuntimeError` de `save_file` | `except BaseAPIException: raise` antes de `except Exception` para propagar `InternalServerError` |

**Ficheiros:** `backend/api/routes/deps.py`, `backend/api/services/auth_service.py`, `backend/api/services/candidate_service.py`, `backend/api/routes/upload.py`

---

### 29. Correções de semântica HTTP

| Local | Antes | Depois |
|-------|-------|--------|
| `routes/results.py` — `POST /run` `ConflictError` | `detail` concatenava dois f-strings sem espaço → texto colado | Espaço e ponto adicionados entre frases |
| `services/candidate_service.py` — `delete_candidate` | `OSError` ao apagar ficheiro levantava `ValidationError` (400) | `InternalServerError` (500) — falha de disco não é erro de validação de input |

**Ficheiros:** `backend/api/routes/results.py`, `backend/api/services/candidate_service.py`

---

### 30. Smoke tests para rotas de autenticação

| Cenário | Resultado esperado |
|---------|--------------------|
| `POST /api/auth/register` — novo utilizador | 201 Created |
| `POST /api/auth/register` — email duplicado | 400 com `error_code` |
| `POST /api/auth/register` — email malformado | 422 |
| `POST /api/auth/login` — credenciais válidas | 200 com `access_token` |
| `POST /api/auth/login` — password errada | 401 com `error_code` |
| `POST /api/auth/login` — email desconhecido | 401 (prevenção de user enumeration) |
| `POST /api/auth/logout` — token válido | 204 No Content |
| `POST /api/auth/logout` — token inválido | 401 com `error_code` |

**Ficheiro:** `backend/api/tests/integration/routes/test_auth_routes.py`

---

### 31. Outras melhorias desta ronda

- **`cli.py`:** corrigido mojibake no banner de arranque (`â€"` → `--`).
- **`docs/TESTING.md`:** secção adicionada a explicar o comportamento de `@lru_cache` em `get_settings()` e como usar `get_settings.cache_clear()` em testes que alterem variáveis de ambiente.
- **`backend/api/tests/conftest.py`:** testes HTTP que precisam de utilizador registado antes do login devem pedir a fixture `test_user` (parâmetro ou `@pytest.mark.usefixtures("test_user")`) quando usam `auth_header` / `jwt_token`.
- **`backend/api/tests/integration/routes/test_upload_results_routes.py`:** corrigido setup (utilizador A registado antes de tokens) para eliminar ERRORs de login 401 em testes 403/smoke.

---

### 32. Verificação (5.ª ronda)

- Suite: `pytest backend/api/tests/` — **80 / 80 testes a passar** (+8 smoke tests de autenticação).

---

# Sexta ronda — Hardening de segurança (críticos e altos)

Alterações aplicadas após uma revisão focada em segurança antes de deployment: refresh tokens, upload DoS, headers de segurança, CSV injection, e robustez de workflows.

---

## 33. Refresh token revocation aplicada no refresh (crítico)

| Antes | Depois |
|-------|--------|
| `refresh_access_token()` validava apenas blacklist + JWT decode; ignorava `RefreshToken.is_revoked` e `expires_at` em BD | `refresh_access_token()` valida que o refresh token existe na BD, **não está revogado** e **não expirou** antes de emitir novo access token |

**Ficheiro:** `backend/api/services/auth_service.py`

Notas:
- Em SQLite, `expires_at` pode vir como datetime naive; a comparação foi normalizada para UTC para evitar `TypeError`.

---

## 34. Blacklist TTL correto (alto)

| Antes | Depois |
|-------|--------|
| `_revoke_token()` colocava **todos** os tokens na blacklist com TTL igual ao refresh token (dias) | `_revoke_token(token_hash, expires_at=...)` recebe expiração explícita; logout revoga access token com TTL curto (1h) e refresh token com TTL longo (7d) |

**Ficheiro:** `backend/api/services/auth_service.py`

---

## 35. Upload hardening — reduzir risco de DoS (crítico/alto)

| Antes | Depois |
|-------|--------|
| `CandidateService.validate_file()` fazia `file.file.read()` (carregava tudo para RAM) | Leitura em chunks (1MB) com abort ao ultrapassar o limite configurado; limite vem de `Settings.get_max_file_size_bytes()` |

**Ficheiro:** `backend/api/services/candidate_service.py`

Notas:
- A mitigação completa também depende de limites a nível de proxy/ingress (body size), mas o service já não faz `read()` ilimitado.

---

## 36. MIME sniffing mais eficiente (alto)

| Antes | Depois |
|-------|--------|
| `python-magic` analisava o buffer completo | `magic.from_buffer(file_bytes[:8192])` (8KB) — reduz custo e mantém deteção |

**Ficheiro:** `backend/api/utils/validators.py`

---

## 37. Security headers adicionados (alto)

| Antes | Depois |
|-------|--------|
| A API só configurava CORS; headers de segurança eram ausentes (testes documentavam ausência) | Middleware adiciona headers essenciais: `X-Content-Type-Options=nosniff`, `X-Frame-Options=DENY`, `Referrer-Policy=same-origin` |

**Ficheiro:** `backend/api/main.py`

**Teste atualizado:** `backend/api/tests/integration/security/test_security.py` passou a validar presença destes headers.

---

## 38. CSV formula injection mitigado (alto)

| Antes | Depois |
|-------|--------|
| `ReportGenerator.save_csv()` exportava células começando com `= + - @` sem escape | Células perigosas são prefixadas com `'` para evitar execução de fórmulas em Excel/Sheets |

**Ficheiro:** `backend/reports/reporter.py`

**Teste atualizado:** `backend/api/tests/integration/security/test_security.py`

---

## 39. Workflow deploy — correção de smoke test (alto)

| Antes | Depois |
|-------|--------|
| `deploy.yml` tinha `expect Exception as e:` e indentação incorreta no snippet python | Corrigido para `except Exception as e:` e indentação consistente |

**Ficheiro:** `.github/workflows/deploy.yml`

---

## 40. Verificação (6.ª ronda)

- Suite: `pytest backend/api/tests/` — **238 / 238 testes a passar**.

## Referência rápida de ficheiros tocados (todas as rondas)

**1.ª ronda**
- `backend/api/db/database.py`
- `backend/api/main.py`
- `backend/api/config.py`
- `backend/api/scoring_config.py`
- `backend/api/models/schemas.py`
- `backend/api/routes/deps.py` (novo)
- `backend/api/routes/processes.py`
- `backend/api/routes/upload.py`
- `backend/api/routes/results.py`
- `backend/api/services/screening_service.py`
- `backend/api/services/report_service.py`
- `backend/api/services/results_query.py` (novo)
- `backend/scoring/scorer.py`
- `backend/reports/reporter.py`
- `backend/api/tests/unit/services/test_screening_service.py`

**2.ª ronda**
- `backend/api/main.py`
- `backend/api/routes/processes.py`
- `backend/api/routes/results.py`
- `backend/api/models/schemas.py`
- `backend/api/utils/errors.py`
- `backend/api/tests/integration/routes/conftest.py`
- `backend/api/tests/integration/routes/test_process_routes.py`
- `backend/api/tests/integration/routes/test_upload_results_routes.py`

**3.ª ronda**
- `backend/api/services/screening_service.py`
- `backend/api/routes/results.py`
- `backend/api/tests/unit/services/test_screening_service.py`
- `backend/api/tests/conftest.py`

**4.ª ronda**
- `backend/api/services/screening_service.py`
- `backend/api/services/process_service.py`
- `backend/api/services/candidate_service.py`
- `backend/api/tests/unit/services/test_screening_service.py`

**5.ª ronda**
- `backend/api/services/process_service.py`
- `backend/api/main.py`
- `backend/api/config.py`
- `backend/api/routes/deps.py`
- `backend/api/routes/results.py`
- `backend/api/routes/upload.py`
- `backend/api/services/auth_service.py`
- `backend/api/services/candidate_service.py`
- `backend/api/tests/integration/routes/test_auth_routes.py`
- `backend/api/tests/integration/routes/test_upload_results_routes.py`
- `backend/api/tests/conftest.py`
- `backend/api/cli.py`
- `docs/TESTING.md`

---

## Sugestões para outros documentos

- **Deploy / produção:** mencionar `DATABASE_URL`, `JWT_SECRET_KEY` obrigatórios e fortes; migrações Alembic em vez de `create_tables`; configurar `STUCK_PROCESS_TIMEOUT_MINUTES` para o tempo máximo aceitável do pipeline.
- **API / erros:** documentar corpo JSON `{ "detail", "error_code" }` para todas as `BaseAPIException`; clientes devem esperar sempre este contrato em todos os endpoints incluindo `/run`, `/results` e `/upload`. Erros de BD respondem agora com **500**, não 400.
- **Resultados:** `summary.strong_matches` / `potential_matches` / `weak_matches` alinhados com as etiquetas do scorer (`MATCH_CATEGORIES`).
- **Estado `cancelled`:** valor canónico é `"cancelled"` (duplo `l`) em toda a stack (ORM, schema, OpenAPI).
- **Screening em background:** estado `processing` ao entrar no runner é tratado como idempotente (retorno silencioso). Qualquer outro estado inesperado lança `ConflictError`. Falha de **todos** os candidatos no loop deixa o processo em **`failed`** com `error_message` genérica (sem detalhe técnico); o detalhe completo encontra-se apenas nos logs. Processos órfãos são recuperados automaticamente no startup via `reconcile_stuck_processes`.
- **Testes:** inventário actual (80 testes), estrutura de pastas, cobertura e **roadmap / fluxos em falta** estão em [`docs/TESTING.md`](TESTING.md). Usar `get_settings.cache_clear()` em testes que alterem variáveis de ambiente. Em testes HTTP, garantir registo do utilizador antes do login (`test_user` ou `usefixtures`).
