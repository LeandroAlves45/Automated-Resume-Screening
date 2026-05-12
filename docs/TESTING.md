# Testing Guide — Automated Resume Screener v2.0

## 🚀 Quick Start: Instalar pytest

### 1. Instalar Dependências Principais
```bash
pip install -r requirements.txt
```

### 2. Instalar Dependências de Testes
```bash
pip install -r requirements-dev.txt
```

### 3. Download do Modelo spaCy (necessário para screening tests)
```bash
python -m spacy download en_core_web_sm
```

---

## 📋 Estrutura de Testes (estado real do repositório)

```
backend/api/tests/
├── conftest.py                    # BD in-memory, client, test_user, auth_header, test_process, serviços, etc.
├── helpers/
│   └── assertions.py
├── integration/
│   └── routes/
│       ├── conftest.py            # autouse: override de get_nlp_model (TestClient não corre lifespan)
│       ├── test_auth_routes.py    # register, login, logout (HTTP)
│       ├── test_process_routes.py # GET processo, 403, 404, serialização cancelled
│       └── test_upload_results_routes.py  # POST /run, GET /results (smoke + 403/404/409)
└── unit/
    └── services/                  # única pasta unitária implementada de momento
        ├── test_auth_service.py
        ├── test_candidate_service.py
        ├── test_process_service.py
        ├── test_report_service.py
        └── test_screening_service.py
```

**Pastas ainda não existentes na árvore** (mencionadas em sprints antigos ou planeadas): `unit/utils/`, `integration/` fora de `routes/`, `security/`, `performance/`. Podem ser criadas quando houver testes.

**Inventário atual (80 testes, `pytest --collect-only`):**

| Área | Ficheiro(s) | O que cobre |
|------|-------------|-------------|
| **Integração HTTP** | [`test_auth_routes.py`](../backend/api/tests/integration/routes/test_auth_routes.py) | `POST /api/auth/register`, `login`, `logout` — 201/400/422/401/204 e contrato `error_code`. |
| | [`test_process_routes.py`](../backend/api/tests/integration/routes/test_process_routes.py) | `GET /api/processes/{id}` — 200 owner, 403 outro utilizador, 404; processo `cancelled` no JSON. |
| | [`test_upload_results_routes.py`](../backend/api/tests/integration/routes/test_upload_results_routes.py) | `POST .../run` e `GET .../results` — 404, 403, 409; resultados para processo em `created`. |
| **Unit — Auth** | `test_auth_service.py` | Registo, login, refresh, logout, `get_current_user` (async). |
| **Unit — Processos** | `test_process_service.py` | CRUD/estados, paginação, `mark_*`, cancelamento. |
| **Unit — Candidatos** | `test_candidate_service.py` | Validação de ficheiro (com mocks a `validate_upload_file`), `save_file`, listagem. |
| **Unit — Relatórios** | `test_report_service.py` | Export CSV/JSON/TXT e ownership (serviço, não rotas HTTP). |
| **Unit — Screening** | `test_screening_service.py` | Pipeline `run`, idempotência `processing`, `ConflictError`, sanitização de erro, resultados/resumo. |

**Cobertura:** o alvo documentado é **≥ 70%** em `backend/api` com `--cov-fail-under=70`. O valor exacto depende da máquina; para ver **linhas em falta** (útil para `utils/validators.py`, `utils/errors.py`, rotas sem testes HTTP):

```bash
pytest backend/api/tests/ --cov=backend/api --cov-report=term-missing
```

---

## Testes em falta (roadmap) e fluxo da aplicação

O fluxo de negócio da API v2, do ponto de vista de um cliente, é em grande parte:

```mermaid
flowchart LR
  register[Register]
  login[Login]
  create[POST process]
  upload[POST upload]
  run[POST run]
  results[GET results]
  export[GET export]
  register --> login --> create --> upload --> run --> results --> export
```

Os testes **já** cobrem pedaços deste fluxo de forma isolada (login/register; criar processo via fixtures; `run`/`results` sem upload real). O que falta, em geral, é fechar o ciclo **HTTP** ou **unitário puro** nas camadas utilitárias.

### 1. Unitário — `utils/validators.py` (recomendado)

| Situação | Porquê |
|----------|--------|
| Não existe `unit/utils/test_validators.py` | A lógica de extensão, tamanho e MIME (`python-magic`) está em [`backend/api/utils/validators.py`](../backend/api/utils/validators.py). `test_candidate_service.py` **moca** `validate_upload_file` nalguns casos, logo a cobertura real do módulo pode ser baixa. |

**Fluxo para implementar:** testes **sem** FastAPI — importar funções (`validate_file_extension`, `validate_upload_file`, etc.), passar `bytes` e nomes de ficheiro em memória, assert das exceções `BaseAPIException` esperadas. Não precisam de `client` nem de `test_db`, salvo se quiseres reutilizar `settings` para limites de tamanho.

### 2. Unitário — `utils/errors.py` (opcional)

| Situação | Porquê |
|----------|--------|
| Ficheiro placeholder `unit/utils/test_errors.py` vazio ou inexistente | Garantir que subclasses de `BaseAPIException` têm `status_code` / `error_code` correctos é sobretudo **regressão**; grande parte já é exercida via serviços e integração. |

**Fluxo:** instanciar 2–3 exceções (ex.: `NotFoundError`, `ConflictError`, `InternalServerError`) e fazer `assert` nos atributos. Sem BD.

### 3. Integração HTTP — `POST /api/processes/{id}/upload`

| Situação | Porquê |
|----------|--------|
| Nenhum teste multipart na pasta `integration/routes/` | Valida o contrato real de upload, `CandidateService.save_file` e disco sob `storage_path` de teste. |

**Fluxo do projecto para o teste:**

1. `test_user` + `auth_header` (ou `usefixtures`) — utilizador registado e token.
2. `POST /api/processes` com `title` + `jd_text` (mín. 50 chars) — processo em `created`.
3. Opcional: se a rota de upload exigir `files_uploaded`, chamar `ProcessService.mark_files_uploaded` via BD/serviço **ou** endpoint interno se existir; hoje o fluxo típico é transição de estado após uploads bem-sucedidos (ver [`upload.py`](../backend/api/routes/upload.py) e [`candidate_service.py`](../backend/api/services/candidate_service.py)).
4. `POST /api/processes/{id}/upload` com `files` multipart (`UploadFile`) — ficheiro `.txt` pequeno ou PDF mínimo válido.
5. Assert: `UploadResponse`, ficheiros aceites/rejeitados conforme o contrato.

**Fixtures:** [`conftest.py`](../backend/api/tests/conftest.py) (`client`, `test_user`, `auth_header`, `test_process`); garantir `override_nlp` via [`integration/routes/conftest.py`](../backend/api/tests/integration/routes/conftest.py) se o módulo de rotas de upload injetar dependências partilhadas com screening.

### 4. Integração HTTP — `GET /api/processes` (lista)

| Situação | Porquê |
|----------|--------|
| Lista paginada só coberta ao nível de **serviço** (`test_process_service.py`) | Falta validar serialização HTTP (`ProcessListResponse`) e query params. |

**Fluxo:** login → `GET /api/processes?skip=&limit=` com `auth_header` → assert estrutura JSON (lista, totais se existirem).

### 5. Integração HTTP — `POST /api/auth/refresh`

| Situação | Porquê |
|----------|--------|
| Refresh está coberto em **serviço** (`test_auth_service.py`), não na rota | Garante que o handler FastAPI e o schema batem certo com o cliente. |

**Fluxo:** `POST /api/auth/login` — guardar `refresh_token` do JSON (a resposta HTTP pode incluir campos extra além do `TokenResponse` mínimo, conforme implementação) → `POST /api/auth/refresh` com corpo `{ "refresh_token": "..." }` → 200 e novo `access_token`; caso negativo com token inválido → 401 + `error_code`.

### 6. Integração HTTP — `GET .../export/{csv,json,txt}`

| Situação | Porquê |
|----------|--------|
| Export está coberto no **ReportService** unitário, não nas rotas [`results.py`](../backend/api/routes/results.py) | Valida headers `Content-Disposition`, status 400/403/404 quando processo não `completed`, etc. |

**Fluxo:** preparar processo **completed** com `Result` na BD (padrão semelhante a `test_process_completed` / dados ORM em [`conftest.py`](../backend/api/tests/conftest.py)) → `GET` export com `auth_header` do owner → 200 e bytes/JSON/texto; utilizador B → 403.

### 7. Integração / E2E — screening completo após `POST /run`

| Situação | Porquê |
|----------|--------|
| Smoke actual chama `run` mas não garante pipeline até `completed` com CVs reais | Útil como teste **lento** ou marcado `@pytest.mark.slow`. |

**Fluxo:** criar processo → `mark_files_uploaded` (ou upload HTTP) → anexar `Candidate` com `raw_text` ou ficheiro que o parser aceite → `POST .../run` (FastAPI `BackgroundTasks` executa após a resposta no `TestClient`) → polling `GET .../results` até `completed` ou `failed`, ou assert intermédio consoante o tempo aceitável em CI.

**Nota:** exige modelo spaCy e possivelmente ficheiros de exemplo; o [`integration/routes/conftest.py`](../backend/api/tests/integration/routes/conftest.py) já injeta `nlp_model` nas rotas.

### 8. Pastas `security/` e `performance/`

Referência histórica em sprints; **não há ficheiros** nestas pastas no estado actual. Quando existirem, usar os marcadores `security` e `load` definidos em `pytest.ini`.

---

## ▶️ Como Executar Testes

### Executar TODOS os testes com coverage
```bash
pytest backend/api/tests/ -v --cov=backend/api --cov-fail-under=70 --cov-report=html
```

**O que isto faz:**
- `-v`: Verbose (mostra todos os testes)
- `--cov=backend/api`: Medir cobertura no módulo backend/api
- `--cov-fail-under=70`: Falhar se cobertura < 70%
- `--cov-report=html`: Gerar relatório HTML em `htmlcov/index.html`

**Depois abrir coverage report:**
```bash
# Windows
start htmlcov/index.html

# macOS/Linux
open htmlcov/index.html
```

---

### Executar apenas testes UNITÁRIOS
```bash
pytest backend/api/tests/unit/ -v
```

### Executar apenas testes de INTEGRAÇÃO
```bash
pytest backend/api/tests/integration/ -v
```

### Executar apenas testes de SEGURANÇA
```bash
pytest backend/api/tests/security/ -v
```

### Executar um ficheiro específico
```bash
pytest backend/api/tests/unit/services/test_auth_service.py -v
```

### Executar um teste específico
```bash
pytest backend/api/tests/unit/services/test_auth_service.py::test_register_user_success -v
```

### Executar testes com padrão de nome
```bash
pytest backend/api/tests/ -k "auth" -v
```

---

## 🔄 Executar Testes em Paralelo (mais rápido)

```bash
pytest backend/api/tests/ -v --cov=backend/api -n auto
```

**O que `-n auto` faz:**
- Usa todos os cores disponíveis
- Executa múltiplos testes simultaneamente
- ~3-4x mais rápido em máquinas modernas

---

## 📊 Gerar Relatórios

### Coverage Report (HTML)
```bash
pytest backend/api/tests/ --cov=backend/api --cov-report=html
open htmlcov/index.html
```

### JUnit XML (para CI/CD)
```bash
pytest backend/api/tests/ --junit-xml=tests-report.xml
```

### Coverage em Terminal (sem HTML)
```bash
pytest backend/api/tests/ --cov=backend/api --cov-report=term-missing
```

---

## 🎯 Markadores de Testes (para executar categorias específicas)

```bash
# Apenas unit tests
pytest -m unit backend/api/tests/ -v

# Apenas security tests
pytest -m security backend/api/tests/ -v

# Tudo EXCETO load tests (rápido)
pytest -m "not load" backend/api/tests/ -v

# Unit OR Integration (não security)
pytest -m "unit or integration" backend/api/tests/ -v
```

---

## 🔍 Debug & Troubleshooting

### Ver logs de teste
```bash
pytest backend/api/tests/ -v --log-cli-level=DEBUG
```

### Parar no primeiro erro
```bash
pytest backend/api/tests/ -x
```

### Parar após N falhas
```bash
pytest backend/api/tests/ --maxfail=3
```

### Mostrar print statements (normalmente supprimidos)
```bash
pytest backend/api/tests/ -s
```

### Mostrar variáveis locais em tracebacks
```bash
pytest backend/api/tests/ -l
```

---

## ⚙️ Configuração do pytest

Tudo configurado em **`pytest.ini`:**

```ini
[pytest]
testpaths = backend/api/tests
python_files = test_*.py *_test.py
addopts = -v --tb=short

markers =
    unit: Unit tests
    integration: Integration tests
    security: Security tests
    load: Performance tests
    asyncio: Async tests
```

---

## 🔐 Vulnerabilidades Corrigidas

As seguintes vulnerabilidades foram **PATCHED**:

| Pacote | Versão Anterior | Versão Nova | Vulnerabilidades Corrigidas |
|--------|-----------------|-------------|---------------------------|
| uvicorn | 0.29.0 | 0.32.0 | Log injection, HTTP response splitting |
| python-multipart | 0.0.9 | 0.0.10 | DoS attacks, arbitrary file write |
| pydantic | 2.0.0 | 2.8.0 | Infinite loop, ReDoS |
| python-jose | 3.3.0 | 3.3.1 | Algorithm confusion, timing attacks |
| pdfminer.six | 20221105 | 20240706 | Pickle deserialization, RCE |
| scikit-learn | 1.3.0 | 1.5.1 | Data leakage |
| python-dotenv | 1.0.0 | 1.0.1 | Symlink following |

**Instalar atualizações:**
```bash
pip install --upgrade -r requirements.txt
```

---

## 📝 Fluxo Típico de Testes (CI/CD)

```bash
# 1. Instalar dependências
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm

# 2. Executar linting
ruff check backend/

# 3. Executar type checks
mypy backend/

# 4. Executar testes COM coverage
pytest backend/api/tests/ \
  -v \
  --cov=backend/api \
  --cov-fail-under=70 \
  --cov-report=xml \
  --junit-xml=tests-report.xml

# 5. Enviar resultados (opcional)
# curl -X POST -F "coverage=@coverage.xml" https://codecov.io/upload
```

---

## 💡 Boas Práticas

1. **Rodar testes localmente antes de commit**
   ```bash
   pytest backend/api/tests/ -v --cov=backend/api
   ```

2. **Usar fixtures (não hardcode)**
   - Todas as fixtures estão em `conftest.py`
   - Importar e usar nos testes: `def test_foo(test_user, test_db, ...)`

3. **Naming de testes** (padrão Arrange-Act-Assert)
   ```python
   def test_register_user_success_returns_user_id(auth_service):
       # ARRANGE
       email = "test@example.com"
       password = "SecurePassword123!"
       
       # ACT
       user = auth_service.register(email, password)
       
       # ASSERT
       assert user.email == email
   ```

4. **Não mockar sem razão**
   - Testes unitários devem usar BD real (in-memory SQLite)
   - Só mockar dependências externas (APIs, serviços)

5. **Cobertura alvo: 70%+**
   ```bash
   pytest --cov=backend/api --cov-fail-under=70
   ```

---

## 🚨 Erros Comuns

### ImportError: No module named 'pytest'
```bash
pip install -r requirements-dev.txt
```

### RuntimeError: event loop is closed
→ Já está resolvido no `pytest.ini` com `asyncio_mode = auto`

### FileNotFoundError: 'en_core_web_sm'
```bash
python -m spacy download en_core_web_sm
```

### Database is locked (SQLite)
→ Usar `:memory:` (já configurado em `conftest.py`)

### Testes falham após alterar variáveis de ambiente / Settings

`get_settings()` usa `@lru_cache`, por isso a instância de `Settings` é
partilhada entre testes na mesma sessão. Se um teste alterar `os.environ`
ou precisar de settings diferentes, é necessário limpar o cache antes e
depois do teste:

```python
from backend.api.config import get_settings

@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
```

Ou, para um único teste:

```python
def test_something_with_custom_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    # ... teste ...
    get_settings.cache_clear()  # limpar no final para não afetar outros testes
```

Alternativamente, a fixture `settings` em `conftest.py` tem scope `session`,
o que significa que corre apenas uma vez. Testes que precisem de settings
diferentes devem criar uma instância de `Settings` directamente em vez de
usar `get_settings()`.

---

## 📚 Recursos

- **Pytest Docs**: https://docs.pytest.org/
- **pytest-asyncio**: https://pytest-asyncio.readthedocs.io/
- **pytest-cov**: https://pytest-cov.readthedocs.io/

---

**Guia de testes v2.0** — inventário alinhado com a árvore actual (`pytest` recolhe **80** testes). Para roadmap e fluxos em falta, ver secção *Testes em falta (roadmap) e fluxo da aplicação* acima.
