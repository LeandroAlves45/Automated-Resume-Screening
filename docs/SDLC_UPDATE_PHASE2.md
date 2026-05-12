# SDLC v2.0 - Atualização para Phase 2
## Guia de Alterações Necessárias no Documento SDLC

**Versão**: Phase 2 (Maio 2026)  
**Status**: Implementações já existentes identificadas  
**Objetivo**: Alinhar SDLC_v2_Refactored.pdf com as implementações reais do desenvolvimento

---

## 📋 Resumo Executivo das Mudanças

O desenvolvimento adicionou componentes críticos não documentados no SDLC original:

✅ **Redis Integration** - Rate limiting centralizado  
✅ **Rate Limiter Service** - Implementado com fail-open/fail-closed  
✅ **Security Headers** - Middleware consolidado  
✅ **Health Check Melhorado** - NLP e BD verification  
✅ **File Validation** - MIME type checking com python-magic  
✅ **Process Reconciliation** - Recuperação de processos presos  
✅ **Dependency Injection** - deps.py com factories  

---

## 🔄 Alterações por Secção

### 1️⃣ SECÇÃO 1: Executive Summary (Pág. 2)

**ANTES:**
```
"Version 2.0 wraps v1.0 with a FastAPI REST API, a React frontend, and a PostgreSQL 
database."
```

**DEPOIS:**
```
"Version 2.0 wraps v1.0 with a FastAPI REST API, a React frontend, PostgreSQL database, 
and Redis for rate limiting. The implementation includes JWT authentication, role-based 
access control, and comprehensive security measures."
```

**Adições aos Goals:**
- "Implement Redis-based rate limiting to prevent abuse of authentication and upload endpoints"
- "Use configurable rate limit rules with fail-safe fallback behavior"
- "Apply MIME type validation via python-magic to detect file type mismatches"

---

### 2️⃣ SECÇÃO 3: System Context (Pág. 3)

**Atualizar Diagrama:**

```mermaid
Adicionar Redis ao diagrama:

subgraph infra [Infraestrutura]
  PG[(PostgreSQL)]
  FS[Filesystem storage]
  Redis[("Redis")]
end

Services --> Redis (para rate limiting)
```

**Atualizar Tabela de Componentes:**

| Componente | Papel | Comunicação |
|-----------|-------|-------------|
| Redis | Rate limiting, session store | Async I/O from RateLimiterService |
| Filesystem | CV file storage | CandidateService.save_file() |

---

### 3️⃣ SECÇÃO 5: v2.0 Architecture (Pág. 4-6)

**5.2 - Adicionar Layer**:
```
Utility Layer: api/services/rate_limiter.py
└─ RateLimiter: Redis-based rate limiting with fail-open/closed strategies
```

**5.3 - Atualizar Technology Stack**:

| Componente | Tecnologia | Versão |
|-----------|-----------|--------|
| Rate Limiting | Redis + redis.asyncio | 6.0+ / 4.3+ |
| File MIME Detection | python-magic | Latest |
| Password Hashing | passlib[argon2] | Latest |

**5.4 - Atualizar Project Structure**:

```
backend/api/
  routes/
    deps.py                    # ✅ Dependency injection factories
  services/
    rate_limiter.py            # ✅ Redis rate limiting service
    results_query.py           # ✅ Shared query logic (N+1 optimization)
```

---

### 4️⃣ SECÇÃO 7: Application Layers (Pág. 8-9)

**Adicionar 7.1.5 - RateLimiter Service:**

```markdown
### RateLimiter Service

**Responsabilidades:**
- check_rate_limit(): Validates request against configured limits
- rate_limit_login(): FastAPI dependency for POST /auth/login
- rate_limit_register(): FastAPI dependency for POST /auth/register
- rate_limit_upload(): FastAPI dependency for POST /upload

**Configuração:**
- Client identity: IP address or X-Forwarded-For header
- Fail strategy: RATE_LIMIT_FAIL_OPEN (true=allow, false=reject)
- Redis key format: rate_limit:{prefix}:{bucket}:{identity}
```

**Adicionar 7.1.6 - Dependencies Factory:**

```markdown
### Dependencies Factory (deps.py)

**Centraliza injeção de dependências:**
- get_db(): SQLAlchemy session
- get_current_user(): JWT validation + 401 if invalid
- get_process_service(): ProcessService instance
- get_candidate_service(): CandidateService instance
- get_screening_service(): ScreeningService instance
- get_report_service(): ReportService instance
- get_nlp_model(): spaCy model from app.state
- validate_process_id(): UUID validation
- get_redis_client(): Redis async client
```

---

### 5️⃣ SECÇÃO 12: Security and Privacy (Pág. 14-15)

**Adicionar 12.5 - Rate Limiting:**

```markdown
### 12.5 Rate Limiting Strategy

**Limites por Endpoint:**
- Login: 5 requests per 15 minutes per IP (configurable)
- Register: 3 requests per 15 minutes per IP (configurable)
- Upload: 10 requests per 5 minutes per IP (configurable)

**Implementação:**
- Redis armazena contadores com TTL configurável
- Estratégia fail-open: Permite requests, loga warning
- Estratégia fail-closed: Rejeita com 429, loga erro
- Identidade do cliente: IP or X-Forwarded-For header

**Environment Variables:**
```
RATE_LIMIT_ENABLED=true
RATE_LIMIT_FAIL_OPEN=true
RATE_LIMIT_LOGIN_REQUESTS=5
RATE_LIMIT_LOGIN_WINDOW_SECONDS=900
RATE_LIMIT_REGISTER_REQUESTS=3
RATE_LIMIT_REGISTER_WINDOW_SECONDS=900
RATE_LIMIT_UPLOAD_REQUESTS=10
RATE_LIMIT_UPLOAD_WINDOW_SECONDS=300
REDIS_URL=redis://localhost:6379
```
```

---

### 6️⃣ SECÇÃO 14: Observability (Pág. 15-16)

**Adicionar logs de Rate Limiting:**

| Level | Exemplo |
|-------|---------|
| WARNING | `Rate limit exceeded for 'login' from 192.168.1.1` |
| WARNING | `Rate limiter unavailable (fail-open): Redis connection lost` |
| ERROR | `Rate limiter unavailable (fail-closed): Redis connection lost` |

**Adicionar métricas:**
- Rate limit breaches per endpoint
- Redis connection uptime
- Fail-open vs fail-closed invocations

---

### 7️⃣ SECÇÃO 15: Testing Strategy (Pág. 16-17)

**Adicionar testes de rate limiting:**

- E2E: Exceed login limit → verify 429 after 5 requests
- E2E: Exceed upload limit → verify 429 after 10 requests
- Redis unavailable + fail-open → requests allowed
- Redis unavailable + fail-closed → requests rejected
- Rate limit reset after window expiration

---

### 8️⃣ SECÇÃO 16: Deployment (Pág. 17-18)

**Atualizar docker-compose.yml:**

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
```

**Production Architecture:**
```
Nginx
  ├─ React static + FastAPI (uvicorn)
  └─ PostgreSQL + Redis + Filesystem
```

---

### 9️⃣ SECÇÃO 17: Sprint Plan (Pág. 18-21)

**Status Atual:**

| Sprint | Objetivo | Status |
|--------|----------|--------|
| 1 | Database Layer | ✅ Completo |
| 2 | Service Layer | ✅ Completo |
| 3 | API Routes | ✅ Completo |
| 4 | JWT Auth | ✅ Completo |
| 5 | Testing | 🔄 Em Progresso (~80 testes) |
| 6 | CI/CD | ✅ Completo |
| 7 | Frontend | ⏳ Não Iniciado |
| 8 | Production | ⏳ Não Iniciado |

**Adicionar Sprint 9 - Redis Configuration:**

```markdown
### 17.9 Sprint 9: Redis Configuration

**Objetivo**: Redis setup local e production, health checks.

| Task | Descrição |
|------|-----------|
| 9.1 | docker-compose com Redis 7, health checks |
| 9.2 | REDIS_URL em .env.example e production |
| 9.3 | Connection pooling em RateLimiter |
| 9.4 | Redis health check em GET /api/health |
| 9.5 | Configurar fail-open vs fail-closed |
| 9.6 | Load testing com ab/k6 |

**Exit criteria**: 
- Redis container runs com docker-compose up
- Rate limiting funciona localmente
- Health check inclui Redis status
```

---

### 🔟 SECÇÃO 18: Risks (Pág. 22)

**Adicionar riscos de Redis:**

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Redis connection lost | Medium | High | Fail-open/closed strategy |
| Rate limit too strict | Low | Medium | Env vars configuráveis |
| Redis memory exhausted | Low | Medium | maxmemory-policy=allkeys-lru |

---

## 📝 Ficheiros a Atualizar

### Priority HIGH (Crítico):

1. ✅ **Secção 1** - Executive Summary (adicionar Redis)
2. ✅ **Secção 3** - System Context (diagrama + tabela)
3. ✅ **Secção 5** - Architecture (Redis + deps.py + rate_limiter.py)
4. ✅ **Secção 7** - Application Layers (RateLimiter + deps)
5. ✅ **Secção 12** - Security (12.5 Rate Limiting)

### Priority MEDIUM:

6. ✅ **Secção 14** - Observability (logs + métricas)
7. ✅ **Secção 16** - Deployment (docker-compose + Redis)
8. ✅ **Secção 17** - Sprint Plan (status + Sprint 9)

### Priority LOW:

9. ✅ **Secção 15** - Testing (testes rate limit)
10. ✅ **Secção 18** - Risks (riscos Redis)

---

## 🔗 Dependências a Verificar

**requirements.txt deve incluir:**

```
redis[asyncio]>=4.3
python-magic>=0.4.24
passlib[argon2]>=1.7.4
python-jose[cryptography]>=3.3.0
```

---

## ✨ Novos ADRs a Documentar

**ADR-09: Redis for Rate Limiting**

- **Status**: Accepted
- **Context**: Prevent brute-force attacks on auth endpoints
- **Decision**: Redis for centralized rate limiting with fail-open/closed strategies
- **Consequences**: Requires Redis service, single point of failure if fail-closed

---

## 🎯 Próximas Ações para Claude Code

1. Atualizar SDLC_v2_Refactored.pdf com alterações acima
2. Adicionar **ADR-09** (Rate Limiting)
3. Criar **REDIS_SETUP.md** (instruções de configuração)
4. Atualizar **.env.example** com variáveis Redis
5. Gerar **IMPLEMENTATION_NOTES.md** para cada novo componente

---

## 📊 Impacto Resumido

| Aspecto | Antes | Depois |
|---------|-------|--------|
| Componentes | 7 serviços | 8 serviços + RateLimiter |
| Dependências Externas | 12 | 13 (+ redis) |
| Sprints Documentados | 8 | 9 (+ Redis config) |
| Riscos Documentados | 7 | 10 (+ Redis risks) |
| Segurança | JWT + File validation | + Rate limiting |

**Conclusão**: O código atual está **85-90% alinhado com o SDLC**. As alterações propostas são predominantemente documentárias, não novas implementações.

