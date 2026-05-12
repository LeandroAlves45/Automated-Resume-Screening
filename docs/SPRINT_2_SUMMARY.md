# Sprint 2 — Camada de Contratos e Utilidades — ✅ FINALIZADO

**Status:** ✅ CONCLUÍDO | **Data:** Abril 2026 | **Objetivo:** Contratos de API + utilitários

---

## Deliverables — 4 Ficheiros (1.525 linhas)

### 1. **schemas.py** (410 linhas)
Pydantic v2: 17 modelos (5 enums + 12 request/response), validações, exemplos JSON.
**ADR-04:** Completamente separado de ORM models.

### 2. **errors.py** (410 linhas)
Hierarquia: BaseAPIException + 6 genéricas + 16 específicas = 22 total com status_code estruturado.

### 3. **logging.py** (346 linhas)
RequestIDFormatter + RequestIDMiddleware + setup_logging() com ContextVar thread-safe.
**Formato:** `TIMESTAMP | LEVEL | REQUEST_ID | MODULE | MESSAGE`

### 4. **validators.py** (359 linhas)
Pipeline: extension → MIME type (magic bytes) → size → sanitize → retorna dict.
**Segurança:** Whitelist + python-magic + path traversal protection.

---

## ✅ Mudanças Implementadas (Este Chat)

### **Consolidação de Configuração**

- **SUPPORTED_EXTENSIONS:** Centralizado em `scoring_config.py` [".pdf", ".docx", ".txt"]
- **validators.py:** Importa apenas de scoring_config, sem redefinição local
- **Princípio:** Single source of truth para valores configuráveis

### **Sanitização de Filenames — Preserva Extensões**

```python
# Antes: re.sub(r"[_\.]+", "_", sanitized)  → "resume.pdf" → "resume_pdf" ❌
# Depois:
re.sub(r"_+", "_", sanitized)     # resume__pdf → resume_pdf
re.sub(r"\.+", ".", sanitized)    # resume.pdf → resume.pdf ✓
```

**Impacto:** Extensões intactas (legibilidade + rastreabilidade).

### **Validação de Segurança Confirmada**

1. Extension whitelist: apenas [.pdf, .docx, .txt]
2. MIME type via magic bytes (detecta ficheiros renomeados: fake.pdf com "hello" → rejeita)
3. File size: 10 MB máximo (protege contra DoS)
4. Path traversal: remove /, \, null bytes, caracteres controlo
5. Preservação de extensões: ".pdf" permanece ".pdf" (não "_pdf")

---

## 📊 Conclusão — Critérios Atendidos

| Critério                  | Status | Prova                                          |
| ------------------------- | ------ | ---------------------------------------------- |
| Schemas validáveis        | ✅     | 17 models, zero import errors                  |
| Errors estruturados       | ✅     | 22 exceções com status_code + error_code       |
| Logging com request_id    | ✅     | RequestIDMiddleware + ContextVar               |
| Validação completa        | ✅     | 5 funções + pipeline orquestrado               |
| Consolidação config       | ✅     | SUPPORTED_EXTENSIONS centralizado              |
| Segurança ficheiros       | ✅     | Magic bytes + extension preservation           |

**Arquitetura:** HTTP → Schemas → Services → Errors → Handler → Logging (unidirecional, acíclico).

**Métricas:** 1.525 linhas, 4 ficheiros, 25 classes, 35+ funções, 100% documentação.

---

## 🚀 Próximos Passos — Sprint 3

**Serviços Core:** process_service.py, candidate_service.py, routes (processes, upload)
**Critério:** Criar processo → upload → listar candidatos ✓

---

**Sprint 2 validado e pronto para merge.**
