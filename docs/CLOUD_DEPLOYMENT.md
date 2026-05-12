# Deploy na cloud — Automated Resume Screener (backend)

**Objetivo:** correr a API FastAPI na **Render** com **PostgreSQL na Render** (Web Service + Render Postgres), sem alterar o teu ambiente local de desenvolvimento. O browser nunca liga directamente à base de dados — só a API.

**Referências no código:**

- Arranque, CORS e health: [`backend/api/main.py`](../backend/api/main.py)
- Variáveis de ambiente e validação produção: [`backend/api/config.py`](../backend/api/config.py)
- Imagem Docker: [`backend/Dockerfile`](../backend/Dockerfile)
- Compose local (contexto de build): [`backend/docker-compose.yml`](../backend/docker-compose.yml)
- Migrações: [`backend/alembic/`](../backend/alembic/)
- Template de env: [`backend/.env.example`](../backend/.env.example)

---

## 1. O que fica onde

| Componente | Função |
|------------|--------|
| **Render Postgres** | PostgreSQL gerido na Render; a API usa `DATABASE_URL` (interna ou externa, conforme a documentação Render). |
| **Render Web Service** | Corre o contentor da API (Dockerfile) com variáveis de ambiente injectadas no dashboard. |
| **Frontend** (futuro) | Pode ficar noutro host (ex.: Vercel); chama só a URL pública da API (`*.onrender.com` ou domínio customizado). O teu PC pode continuar com `APP_ENV=development` e Docker local. |

Fluxo: **Browser → API (Render) → PostgreSQL (Render Postgres)**.

---

## 2. Render — PostgreSQL

1. No [Render Dashboard](https://dashboard.render.com), cria um **PostgreSQL** (plano conforme a tua conta; existe tier gratuito com limitações documentadas pela Render).
2. Copia a **Internal Database URL** ou **External Database URL** para `DATABASE_URL` no Web Service da API, conforme a Render indicar para ligação serviço-a-serviço vs. ferramentas externas.
3. Garante **SSL** se a connection string o exigir (segue a documentação oficial da Render para Postgres).

---

## 3. Schema da base de dados (Alembic)

Em **produção** (`APP_ENV=production`), a aplicação **não** executa `create_tables()` no arranque; o schema deve ser aplicado com **Alembic**.

1. Com a mesma `DATABASE_URL` que o serviço na Render vai usar, corre (com o ambiente Python do projecto):

   ```bash
   cd backend
   alembic upgrade head
   ```

   O Alembic lê a URL a partir da configuração da app ([`backend/alembic/env.py`](../backend/alembic/env.py)).

2. Em alternativa, na Render podes configurar um **Pre-Deploy Command** (ou correr uma shell one-off com as mesmas env vars) para executar `alembic upgrade head` em cada deploy que altere o schema — consulta a documentação Render para o runtime Docker.

Sem este passo, o arranque pode falhar ao verificar a BD ou as rotas falham por tabelas em falta.

---

## 4. Render — Web Service (API)

### 4.1 Build Docker (importante)

O [`backend/docker-compose.yml`](../backend/docker-compose.yml) faz build com:

- **context:** raiz do repositório `resume-screener` (pasta que contém o pacote Python `backend/`)
- **dockerfile:** `backend/Dockerfile`

O `Dockerfile` faz `COPY . .` sobre esse contexto e arranca com Uvicorn em `backend.api.main:app`.

**Na Render:** cria um **Web Service** com **Docker**, define a **raiz do repositório** como a pasta que contém o pacote `backend/` (normalmente a raiz do repo `resume-screener`) e o caminho do Dockerfile como `backend/Dockerfile`. Se o root do build for só `backend/` sem ajustar o Dockerfile, o módulo `backend.api.main` deixa de resolver.

### 4.2 Porta HTTP (`PORT`)

A Render injecta a variável de ambiente **`PORT`**. O contentor deve escutar nessa porta para o tráfego público chegar ao processo. Se a imagem estiver fixa na porta `8000`, configura na Render o comando de arranque ou ajusta o Dockerfile para usar `${PORT}` — alinha com a documentação Render para serviços Docker.

### 4.3 Armazenamento de uploads

`STORAGE_PATH` (por omissão `./storage` em [`backend/api/config.py`](../backend/api/config.py)) é disco **local ao contentor**. Em cada redeploy esse disco pode ser **recriado**. Para aprendizagem/MVP pode bastar; para ficheiros persistentes, prevê **disco persistente** na Render (planos pagos) ou storage externo.

### 4.4 Redis e rate limiting

Por omissão o rate limiting está desligado (`RATE_LIMIT_ENABLED=false`). Se activares, precisas de um **Redis** acessível e de `REDIS_URL` correcto (ex.: **Render Key Value** ou serviço Redis externo). Ver também [`docs/REDIS_RATE_LIMITING.md`](REDIS_RATE_LIMITING.md).

---

## 5. Variáveis de ambiente (Render Dashboard)

Define no painel do **Web Service** as variáveis abaixo. Os nomes seguem o que o Pydantic Settings lê em [`backend/api/config.py`](../backend/api/config.py) (case-insensitive).

| Variável | Obrigatório em produção | Notas |
|----------|-------------------------|--------|
| `DATABASE_URL` | Sim | URL do **Render Postgres** (ou referência interna que a Render gere); **não** uses o placeholder de desenvolvimento local quando `APP_ENV=production`. |
| `APP_ENV` | Sim | Define `production` para o serviço na cloud. |
| `JWT_SECRET_KEY` | Sim | Em produção: mínimo **32 caracteres** e não pode ser o placeholder de dev. Gera com: `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `LOG_LEVEL` | Recomendado | Ex.: `INFO` ou `WARNING`. |
| `ALLOWED_ORIGINS` | Recomendado | Lista separada por vírgulas dos origins do frontend (ex.: `http://localhost:5173` + URL HTTPS do frontend em produção). |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Opcional | Por omissão 60. |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | Opcional | Por omissão 7. |
| `REDIS_URL` | Se rate limit activo | Por omissão aponta para localhost — na cloud tens de apontar para um Redis real. |
| `RATE_LIMIT_ENABLED` | Opcional | `true` / `false`. |
| `SPACY_MODEL` | Opcional | Por omissão `en_core_web_sm`; o Dockerfile já instala este modelo. |

Não commits ficheiros `.env` com segredos; usa apenas variáveis no dashboard da Render.

---

## 6. Verificação depois do deploy

- **Health check HTTP:** `GET /health` na URL pública do serviço (ex.: `https://<nome>.onrender.com/health`).
- Resposta esperada: JSON com `status`, `database`, `nlp_model`, `environment`, etc. (modelo `HealthResponse`).
- Na Render podes configurar **Health Check Path** para `/health`.

---

## 7. CI/CD e Git (opcional)

Muitos projectos ligam o repositório GitHub à Render e fazem **deploy automático** a cada push na branch escolhida, sem GitHub Actions.

O workflow [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) faz deploy alternativo:

1. **Backend (Render):** `POST` ao **Deploy Hook** do Web Service (secret `RENDER_DEPLOY_HOOK_URL` — copia o URL completo em Render → Web Service → **Deploy Hook**).
2. **Verificação:** espera pelo build e faz `GET` a `{RENDER_SERVICE_URL}/health` (secret `RENDER_SERVICE_URL` = URL base pública **sem** barra final, ex.: `https://teu-api.onrender.com`).
3. **Frontend (Vercel):** mantém os segredos `VERCEL_TOKEN`, `VERCEL_PROJECT_ID`, `VERCEL_ORG_ID`, `VERCEL_DOMAIN`, `VERCEL_DEPLOYMENT_URL` conforme o teu projecto.

Se não usares Actions, podes apagar ou desactivar o workflow; a Render continua a poder fazer deploy só por push ao repo ligado.

---

## 8. Checklist rápido

- [ ] Instância **Render Postgres** criada; `DATABASE_URL` definida no Web Service.
- [ ] `alembic upgrade head` aplicado na base (localmente com a URL de produção ou via Pre-Deploy na Render).
- [ ] **Web Service** Docker: root = repo `resume-screener`, Dockerfile = `backend/Dockerfile`.
- [ ] `APP_ENV=production`, `JWT_SECRET_KEY` forte, `DATABASE_URL` ≠ placeholder local.
- [ ] `ALLOWED_ORIGINS` inclui o(s) origin(s) do frontend que vão chamar a API.
- [ ] Processo escuta em **`PORT`** se a Render o injectar.
- [ ] `GET /health` devolve 200 após o deploy.
- [ ] (Opcional) Redis + `RATE_LIMIT_ENABLED` se precisares de limite de pedidos na cloud.

---

## 9. Tier gratuito e aprendizagem

Os planos **gratuitos** da Render (Web Service e Postgres) têm **limitações** (hibernação do web service após inactividade, quotas, políticas de retenção da base grátis, etc.). São adequados para aprender e para portfólio; consulta sempre a [documentação e pricing](https://render.com/docs) actual.

---

## 10. Desenvolvimento local vs cloud

- **Local:** mantém `APP_ENV=development` (e `.env` com Postgres Docker ou local, como em [`backend/docker-compose.yml`](../backend/docker-compose.yml)).
- **Cloud:** o serviço na **Render** usa `APP_ENV=production` e segredos reais; não precisas de “converter” o teu PC para produção — são ambientes separados.
