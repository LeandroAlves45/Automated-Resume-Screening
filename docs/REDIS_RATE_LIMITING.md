## Redis rate limiting (backend)

Este projeto agora suporta **rate limiting via Redis** (throttling) para reduzir brute force e abuso em endpoints sensíveis.

### O que foi implementado

- **Redis client + rate limiter**: `backend/api/services/rate_limiter.py`
- **Novo erro 429**: `backend/api/utils/errors.py` (`RateLimitExceededError`, `error_code="RATE_LIMITED"`)
- **Endpoints protegidos**:
  - `POST /api/auth/login` (bucket `login`)
  - `POST /api/auth/register` (bucket `register`)
  - `POST /api/processes/{process_id}/upload` (bucket `upload`)

Por default, o rate limiting está **desligado** (para não quebrar dev/local):

- `RATE_LIMIT_ENABLED=false`

### Dependência

O backend usa `redis-py`:

- `requirements.txt` inclui `redis>=6.0.0`

### Como correr Redis localmente

#### Opção A — Docker (recomendado)

```bash
docker run --name ars-redis -p 6379:6379 -d redis:7
```

Para parar/remover:

```bash
docker rm -f ars-redis
```

#### Opção B — Redis local instalado

Se tiveres Redis instalado no sistema, garante que está a correr em `localhost:6379`.

### Como ativar rate limiting

Define estas env vars (em `.env` do backend ou no ambiente do deploy):

```bash
REDIS_URL=redis://localhost:6379/0
RATE_LIMIT_ENABLED=true
RATE_LIMIT_FAIL_OPEN=true
```

#### Configurar limites (opcional)

Os limites são **fixed-window** (por IP) e configuráveis por env var:

```bash
# login
RATE_LIMIT_LOGIN_REQUESTS=10
RATE_LIMIT_LOGIN_WINDOW_SECONDS=60

# register
RATE_LIMIT_REGISTER_REQUESTS=5
RATE_LIMIT_REGISTER_WINDOW_SECONDS=60

# upload
RATE_LIMIT_UPLOAD_REQUESTS=30
RATE_LIMIT_UPLOAD_WINDOW_SECONDS=60
```

### Como funciona (resumo técnico)

- A chave usada no Redis é por **bucket + client identity**:
  - `rate_limit:v1:{bucket}:{identity}`
- `identity` tenta usar `request.client.host` (e fallback para `x-forwarded-for`).
- O contador é incrementado com `INCR`.
- Na primeira request da janela, a key recebe `EXPIRE window_seconds`.
- Quando excede o limite, a API retorna:
  - **HTTP 429**
  - JSON: `{ "detail": "...", "error_code": "RATE_LIMITED" }`

### Testes de integração (com Redis)

Há um teste de integração que valida `429` para login quando o limite é excedido:

- `backend/api/tests/integration/security/test_security.py::TestRateLimitingRedis`

Para correr (com Redis a correr):

```bash
pytest backend/api/tests/integration/security/test_security.py -v
```

Notas:
- O teste usa, por default, `REDIS_URL=redis://localhost:6379/15` para reduzir colisões.
- Se Redis não estiver disponível, o teste faz **skip** (não falha).

### Deploy (Render + Vercel)

- **Backend (Render)**: configura `REDIS_URL` com o URL do Redis provisionado (ex.: Render Key Value ou Redis externo) e ativa `RATE_LIMIT_ENABLED=true`.
- **Frontend (Vercel)**: sem mudança (continuas a usar Bearer token no header).

