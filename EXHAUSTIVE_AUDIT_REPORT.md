# EXHAUSTIVE AUDIT REPORT (eduCore, branch `main`)

Date: 2026-11-21 (generated during automated audit session)
Repository: `/media/amin/data/Machine-Learning-Engineer/09-Volte IQ/eduCore` (git branch: `main`)
Working tree (observed): untracked files include `.venv/`, `CONTINUATION_PLAN.md`, `MODEL_TEST_REPORT.md`, `session-ses_e01f.md`, `test_models_report.py`, `test_full.py` (no committed code diff shown beyond these new/untracked items in the checked status snippet).

---

## 1) Scope & Method (what was done)

This audit covers:
- Repository structure and key config/providers/models/routes/services/tests under `educore/`, `tests/`, root scripts, and `docker-compose.yml`.
- Configuration inspection (`.env.example`, live `.env` present locally), provider/model routing code, API routes, RAG/chat/student/teacher services, teacher AI modules, Celery/app entrypoints.
- Automated unit tests executed: `python -m pytest -q` using `conda env: ml` (`/home/amin/miniconda3/envs/ml/bin/python`).
- Code review for obvious defects and “safe/clear” fixes where applicable.

**Not performed as live end-to-end integration runs in this session** (would require: running Postgres/RabbitMQ/Redis per actual `.env`, running FastAPI/Celery, uploading documents, exercising external APIs with quota/rate limits). Those flows are documented as **external/quota/env-dependent** with recommended reproduction steps.

---

## 2) Configured / Available Models & Providers (from code + config)

### 2.1 Text / chat (LLM)
- **Provider lock (code):** `educore/provider.py` enforces `LLM_PROVIDER` must normalize to `groq`; otherwise raises `ValueError("Text generation is configured for Groq only. Set LLM_PROVIDER=groq.")`. `require_api_key()` requires `GROQ_API_KEY`.
- **Groq client defaults:** `GROQ_BASE_URL=https://api.groq.com/openai/v1` (OpenAI-compatible). Async/sync clients created via `create_async_client/create_sync_client` with timeouts/retries tuned in code.
- **Active chat model (settings):** `settings.active_chat_model` returns `GROQ_CHAT_MODEL` when provider is Groq (`educore/config.py`). Fallback list in generators often uses `[primary, GROQ_FAST_MODEL]` (see `educore/rag/generator.py`, `educore/teacher_ai/core/llm_client.py` patterns).

**From live `.env` (observed):**
- `LLM_PROVIDER=groq`
- `GROQ_API_KEY` set (non-empty in file)
- `GROQ_BASE_URL=https://api.groq.com/openai/v1`
- `GROQ_CHAT_MODEL=openai/gpt-oss-120b`
- `GROQ_FAST_MODEL=openai/gpt-oss-20b`
- `GROQ_WEB_SEARCH_ENABLED=false`
- `OPENROUTER_API_KEY` present (legacy/optional fields in config; text path uses Groq per code)
- `OPENROUTER_ENABLE_WEB_SEARCH=false` (legacy flag noted in comments)

**From `.env.example`:**
- Documents Groq defaults and notes `LLM_PROVIDER=groq`, `GROQ_CHAT_MODEL=groq/llama-3.1-8b-instant` in example (differs from live `.env` in model naming style in that example line) — **configuration/documentation drift risk** (see Findings).

### 2.2 Image / vision
- **Gemini (Google AI Studio):** `GOOGLE_API_KEY`, `GOOGLE_VISION_MODEL` (default in config: `gemini-2.5-flash`), upload/compression limits (`VISION_MAX_UPLOAD_BYTES`, `VISION_MAX_COMPRESSED_BYTES`). Used in student image analysis flows (`educore/services/student_service.py`, routes under `/student/analyze-image`, chat image endpoint in `api/routes.py` referencing image analysis behavior).
- Live `.env`: `GOOGLE_API_KEY` set, `GOOGLE_VISION_MODEL=gemini-2.5-flash`.

### 2.3 Embeddings
- `EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2` (live `.env`), `EMBEDDING_DIMENSIONS=384` in live `.env` (must match model). `educore/rag/embedder.py` / usage implied by config.
- Note: `.env.example` shows `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2`, `EMBEDDING_DIMENSIONS=384` — **drift vs live env** (model name differs).

### 2.4 Other runtime deps (infra)
- Postgres+pgvector: live `.env` points to `localhost:5400`, DB user `postgres`, db `educore` (differs from `docker-compose.yml` defaults `educore/educore` on 5432). **Operational mismatch risk** if following README/docker-compose blindly.
- Celery broker: `amqp://appuser:...@localhost:5672//` (live). Result backend Redis with password placeholder style in file.
- `educore/celery_app.py`, `educore/tasks/ingestion_tasks.py` present.

---

## 3) API Surface (high-level, from `educore/api/routes.py`)

- Health: `GET /api/health`
- Users: create/list/get
- Projects: create/list/delete
- Documents: upload, list, status, delete (Celery task enqueued on upload)
- Chat: sessions CRUD/list/messages, `POST /chat/sessions/{id}/ask` (SSE), image-in-chat endpoint exists (`ask-image` style route present in file review)
- Student: dashboard, practice question, study plan, exams generate/submit, image analysis, exam hints
- Teacher: materials retrieval, assessment records

FastAPI app wiring: `educore/main.py` (CORS allows `http://localhost:8501`, startup logs model/provider info).

---

## 4) Automated Test Results (executed)

Command (from repo root):
- `/home/amin/miniconda3/envs/ml/bin/python -m pytest -q`

Result:
- `28 passed, 2 subtests passed in ~16.20s` (as recorded in this session)

Relevant test files:
- `tests/test_rag_pipeline.py`
- `tests/test_teacher_model_fixes.py`

**Interpretation:** Unit-level behaviors covered by these tests pass. This does **not** prove live Groq/Gemini/Postgres/RabbitMQ/Redis/Celery/Streamlit integrations work in the current machine state.

---

## 5) Findings (issues, risks, and observations)

### F-01 (Documentation/config drift) — Medium
- **Where:** `.env.example` vs live `.env` and vs `README.md` snippets.
- **Examples:**
  - Example shows `GROQ_CHAT_MODEL=groq/llama-3.1-8b-instant`, live uses `openai/gpt-oss-120b`.
  - Example shows `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2`; live uses `paraphrase-multilingual-MiniLM-L12-v2` (dimensions both 384 in observed files but **model identity differs** — retrieval behavior/index compatibility depends on model; changing models requires re-embedding/index strategy awareness).
  - `README.md` still mentions OpenRouter-centric text generation in some places (e.g. “Groq (LLM)”, “OpenRouter LLM streaming generation” in architecture text) while code is Groq-locked — **misleading for operators**.
  - `docker-compose.yml` uses Postgres user/db `educore/educore` on 5432; live `.env` uses `postgres`/`educore_pass` on 5400. README quickstart may not match actual local stack.
- **Root cause:** Config/docs not synchronized with runtime choices.
- **Risk:** Wrong model downloads, failed auth assumptions, wrong DB connection, onboarding errors.
- **Recommended fix (safe):** Update `.env.example` and README to reflect Groq-only flow, current defaults you intend, and document the actual infra ports/credentials used (or align them). (Not applied automatically here to avoid guessing intent.)

### F-02 (Provider module has redundant/odd logic) — Low
- **Where:** `educore/provider.py`
- **Detail:** Logic checks `if provider == "groq": ...` then later `if provider != "groq": raise ...` (unreachable in effect given flow) — harmless but confusing.
- **Root cause:** Incremental edits.
- **Risk:** Low (maintenance confusion).
- **Suggested fix:** Simplify to a single canonical branch for `groq` only.

### F-03 (Teacher AI Groq fallback list defined but not consistently used in all paths) — Low/Medium
- **Where:** `educore/teacher_ai/core/llm_client.py` contains `_FALLBACK_MODELS = ([settings.GROQ_FAST_MODEL])` and related Groq handling; also `educore/rag/generator.py` implements fallback logic. Behavior depends on implementation details.
- **Risk:** Inconsistent fallback behavior between RAG chat and teacher AI modules under 429/errors.
- **Recommended verification:** Add/extend tests or run live smoke tests per model.

### F-04 (Live E2E not executed here: models + all features) — Informational
- **Where:** N/A (environment/process)
- **Detail:** No running FastAPI on the expected port was verified in this session beyond process listing showing other services; no HTTP calls made to eduCore API to validate full flows (ingestion→index→chat→student exams→teacher endpoints→image vision end-to-end). The `test_models_report.py` / `test_full.py` scripts exist but were not executed here.
- **Risk:** Unknown runtime failures (DB migrations not applied to the live 5400 DB, missing tables, Celery worker not consuming `educore.process`, etc.).
- **Recommended reproduction (manual):**
  1. Ensure infra up (Postgres/pgvector, RabbitMQ, Redis) matching `.env` (`DATABASE_URL*`, broker, Redis).
  2. `alembic upgrade head`
  3. Start API: `uvicorn educore.main:app --reload --port 8002` (per live `.env` API_PORT) and Celery worker for queue used by app (`educore.celery_app`, queue `educore.process` per `educore/tasks/ingestion_tasks.py` / `celery_app` config).
  4. Run `python test_models_report.py` (from repo root, with `ml` env) to exercise Groq primary/fallback, embeddings, Gemini image, provider lock — **expect 429/quota/auth errors to be recorded, not to crash the script** (script already catches exceptions).
  5. Run `python test_full.py` against the running API (script uses `API_BASE=http://127.0.0.1:8002/api`) to validate upload/index/chat/student flows — **will fail fast if API/DB/Celery not ready**.

### F-05 (Security hygiene: secrets present in local `.env`) — High (operational)
- **Where:** `/media/amin/data/Machine-Learning-Engineer/09-Volte IQ/eduCore/.env` (local file)
- **Detail:** Contains API keys and DB passwords in plaintext on disk (observed during audit). This is common locally but risky if committed or shared.
- **Root cause:** Local development configuration.
- **Risk:** Credential leakage.
- **Recommended fix:** Ensure `.env` is gitignored (verify `.gitignore`), rotate keys if ever exposed, never paste real keys into reports/logs committed to git. (No key values are included in this report.)

---

## 6) “Test each model individually (handling 429/rate-limit/quota/model-type errors gracefully without aborting the whole suite)”

**Code-level expectations (from review):**
- `educore/rag/generator.py`: retries with backoff on 429/rate limits, falls back across models in some paths, and distinguishes Groq reasoning effort handling. Streaming has fallback behavior.
- `test_models_report.py`: catches all exceptions per test, records `FAIL` with error string, continues to next test — **graceful, non-aborting** behavior by design.

**Gap:** Not re-executed here against live network in a controlled recorded run inside this audit session; recommend running the script and attaching its generated `MODEL_TEST_REPORT.md` output to QA records.

**Models to cover explicitly (per repo):**
1. Groq chat primary: `GROQ_CHAT_MODEL` (`openai/gpt-oss-120b` in live `.env`)
2. Groq chat fallback: `GROQ_FAST_MODEL` (`openai/gpt-oss-20b`)
3. Embeddings: `EMBEDDING_MODEL_NAME` (`paraphrase-multilingual-MiniLM-L12-v2`)
4. Gemini vision: `GOOGLE_VISION_MODEL` (`gemini-2.5-flash`)
5. Provider lock/config sanity (`provider_settings`)

---

## 7) “Test all features and flows…” (checklist mapped to code)

| Area | Code entrypoints (files) | Live E2E in this audit |
|------|---------------------------|------------------------|
| Ingestion (upload, parse/extract/chunk/embed/store, async) | `api/routes.py` upload + `tasks/ingestion_tasks.py` + `services/ingestion_service.py` + `rag/*` | Not run (needs API+Celery+DB+storage) |
| RAG chat (sync/stream SSE, history, citations/validation) | `api/routes.py` chat endpoints, `services/chat_service.py`, `rag/generator.py`, `rag/retriever.py`, `rag/validator.py` | Not run |
| Search/retrieval behavior | `rag/retriever.py`, embeddings | Unit tests cover some pieces? (only RAG pipeline + teacher tests present) |
| Student: dashboard/practice/study plan/exams submit/hints/image | `api/routes.py` student routes, `services/student_service.py`, `student_ai*` | Not run |
| Teacher: materials, assessment records, teacher_ai modules | `api/routes.py` teacher routes, `services/teacher_service.py`, `educore/teacher_ai/*` | Unit tests for some teacher schema/logic exist and pass |
| Image/vision | student analyze image, chat image route | Not run (needs Gemini key/quota) |
| Error/edge/auth/ownership | `api/deps.py`, route checks (`PermissionError` → 403 in many places), validations | Partially reviewable in code; not runtime-tested |

---

## 8) Bugs found and actions taken

### 8.1 Fixed in this audit (safe/clear)
- **None applied** after review: the only automated tests available (`pytest`) pass. I did not change production code because no concrete failing behavior was demonstrated by tests and live E2E was not executed; speculative changes risk regressions.

### 8.2 Recommended code/doc fixes (not applied — require product/ops decisions)
1. **Align `.env.example` and README with Groq-only + real defaults** (models, embedding model, ports, DB credentials) — see F-01.
2. **Simplify `educore/provider.py` branching** — see F-02.
3. **Verify/fix any remaining OpenRouter mentions in user-facing docs** that contradict Groq-only behavior.

---

## 9) External / quota / env / infra issues (separate from code bugs)

Record these when running live tests (template):

- **Groq 429 / rate limit / quota:** model `openai/gpt-oss-120b` or `openai/gpt-oss-20b` — symptom: HTTP 429 from Groq/OpenAI-compatible API; expected graceful handling per `generator.py` retry/fallback paths. Action: wait/retry, reduce load, check Groq console quotas.
- **Gemini 429/quota or auth errors:** `GOOGLE_VISION_MODEL=gemini-2.5-flash` with `GOOGLE_API_KEY` — record HTTP/status and message from `student_service.py` error mapping.
- **Missing/invalid env:** `GROQ_API_KEY` empty, wrong `DATABASE_URL*`, broker/Redis unreachable, `ALEMBIC` not migrated — causes 500s or startup failures.
- **Model/type mismatches:** changing `EMBEDDING_MODEL_NAME` without reindexing can break retrieval dimensions/semantics; `EMBEDDING_DIMENSIONS` must match model output.
- **Infra mismatch:** README `docker-compose.yml` vs live `.env` ports/credentials (5400/postgres vs 5432/educore) — ensure the stack you run matches `.env`.

---

## 10) How to reproduce a full live audit (runbook)

```bash
cd "/media/amin/data/Machine-Learning-Engineer/09-Volte IQ/eduCore"
# 1) Unit tests
/home/amin/miniconda3/envs/ml/bin/python -m pytest -q

# 2) Model smoke (live APIs) — may incur usage; records errors gracefully
/home/amin/miniconda3/envs/ml/bin/python test_models_report.py
# -> writes MODEL_TEST_REPORT.md

# 3) Full API flow smoke (requires API + DB + Celery + Redis/RabbitMQ as configured)
#    Start services per .env, then:
/home/amin/miniconda3/envs/ml/bin/python test_full.py
```

---

## 11) Summary (original static review)

- **Unit tests:** PASS (28 passed, 2 subtests passed).
- **Code/config review:** Identified documentation/config drift (F-01), minor code clarity issue (F-02), and need for live E2E validation (F-04). No definitive runtime code bug proven by failing unit tests in the earlier review; no speculative production code changes applied.

---

## 12) Appendix — Live verification run (2026-11-21, executed in this session)

### 12.1 Local services (observed via `ss` / process listing)
- **Postgres (pgvector)** listening on **`0.0.0.0:5400`** (matches live `.env` `DATABASE_URL*` host/port context). **Reachability:** assumed available if API/DB calls succeed; not separately `pg_isready`-logged here.
- **RabbitMQ** listening on **5672, 15672, 15692** (present).
- **Redis** listening on **6379** (present).
- **eduCore FastAPI (uvicorn)** process observed: running against **`127.0.0.1:8002`** (matches live `.env` `API_PORT=8002`) with log redirection to `storage/logs/api.log` (per running command). **Reachable:** confirmed by successful HTTP calls from `test_full.py` to `http://127.0.0.1:8002/api`.
- **eduCore Celery worker(s)** observed: `celery -A educore.celery_app worker ... -Q educore.process` (both a solo worker PID 209414 and a prefork pool under `.venv` on pts/4). **Inference:** worker process present; document indexing in `test_full.py` completed (doc became `indexed` with chunks), which is consistent with a working worker+broker+DB path for that flow.

Other unrelated services (Notionllm stack on 8000/5400 overlap context aside: this machine shows multiple stacks) were present; the eduCore-specific listeners used by this run are **:5400 (Postgres), :5672/:15672 (RabbitMQ), :6379 (Redis), :8002 (eduCore API)**.

### 12.2 Command: `test_models_report.py` (`/home/amin/miniconda3/envs/ml/bin/python test_models_report.py`)
- **Generated file:** `MODEL_TEST_REPORT.md` (2026-11-21T15:08:32.262062Z). Results:
  - `openai/gpt-oss-120b` (llm, Groq primary): **PASS**, ~**1075.61 ms**.
  - `openai/gpt-oss-20b` (llm, Groq fallback): **PASS**, ~**671.75 ms**.
  - `all-MiniLM-L6-v2` (embedding): **PASS**, ~**5429.77 ms** (sample note indicates 384-dim vectors for 3 texts in that script’s test).
  - `gemini-2.5-flash` (image): **FAIL** — `ClientError: 404 NOT_FOUND`. Message indicates model not available to new users / update guidance (see `MODEL_TEST_REPORT.md` for full text).
  - `provider-lock (Groq-only)` (config): **PASS** (Groq-only lock confirmed by script output).
- **Errors handled gracefully:** script continued after Gemini failure (no early abort). No 429/quota observed in this run for Groq/embedding steps shown.

### 12.3 Command: `test_full.py` (`/home/amin/miniconda3/envs/ml/bin/python test_full.py`)
- **Generated file:** `FULL_TEST_REPORT.md` (2026-11-21T15:09:31.480615Z), run tag `20261121150903`, API `http://127.0.0.1:8002/api`.
- **Totals:** **22 tests, 18 passed, 4 failed, 0 info**.
- **Key pass examples (latencies from report):** Groq primary LLM **~790 ms**, Groq fallback **~606 ms**, embedding **~7015 ms**. RAG chat (ar/en study + en hint) passed with citations (latencies ~1.3s–2.0s). Upload → indexing succeeded (doc indexed, **2 chunks**). Teacher assessment records and student dashboard returned successfully for the test scope. Authorization/ownership negative checks returned expected HTTP codes (404/400/409) as recorded.
- **Key failures / notable errors (concrete):**
  1. **Gemini image model (model smoke inside full script context):** **FAIL** — 404 NOT_FOUND for `models/gemini-2.5-flash` (same class of issue as `test_models_report.py`).
  2. **Exam generation:** **FAIL** — “No topics discovered”.
  3. **Study plan:** **FAIL** — HTTP **404** with detail: *“Index the teacher's course files before creating a study plan.”*
  4. **Image analysis (API flow):** **FAIL** (856.61 ms) — HTTP **429**: *“Gemini quota or request limit reached. Check Google AI Studio quota and try again later.”*
- **RAG chat evaluation entries (from report):** 4 turns recorded (AR study, EN study, AR follow-up study, EN hint) with answer lengths and citation counts shown in `FULL_TEST_REPORT.md`.

### 12.4 Code bugs vs external/quota/service issues (clear separation)
- **External/quota/service/config (not “code logic bugs” to fix blindly here):**
  - Gemini model availability: `gemini-2.5-flash` returns **404 NOT_FOUND** in these runs (provider/model deprecation/availability for this API/key context). Requires aligning `GOOGLE_VISION_MODEL` (and any related client usage assumptions) to a currently allowed model per Google AI Studio guidance.
  - Gemini image analysis via API returned **HTTP 429** (quota/rate). This is an external quota/service condition; application mapped it to a 429 response (observed). Retrying later or changing quota/billing is outside repo code.
  - Groq calls in these runs succeeded for the tested prompts (no 429 shown); if 429 occurs later, existing retry/fallback behavior in `educore/rag/generator.py` / related paths is the designed mitigation (still external throttling).
- **Potential product/code or data/flow issues to verify (not proven fixed):**
  - **Exam generation “No topics discovered”** and **Study plan 404** (“Index the teacher's course files before creating a study plan.”) after a TXT upload that was indexed (2 chunks) suggest either (a) topic extraction requires different content/format expectations, (b) study-plan prerequisites differ from “any indexed doc” in this flow, or (c) a regression/logic mismatch. These need reproduction against the exact `test_full.py` data path and service logs (`storage/logs/api.log`, Celery logs) to classify as bug vs expected precondition messaging. **No production code was changed** for this in the live run session except documentation updates to this report.
- **No unit-test regressions:** `pytest` still **28 passed, 2 subtests passed** earlier in this session (re-run not repeated after doc edits except implicit sanity; doc-only change).

### 12.5 Artifacts produced/updated in this live run
- `MODEL_TEST_REPORT.md` (generated/refreshed by `test_models_report.py`)
- `FULL_TEST_REPORT.md` (generated by `test_full.py`)
- `EXHAUSTIVE_AUDIT_REPORT.md` (this file — appended live appendix)

### 12.6 Recommended next concrete steps (minimal, evidence-based)
1. **Gemini:** set `GOOGLE_VISION_MODEL` to a currently available model (e.g. per Google’s current recommendations) in `.env`/config defaults, re-run `test_models_report.py` and the image-related parts of `test_full.py`; treat 404 as config/model-availability fix, 429 as external quota.
2. **Exams/study plan after indexing:** capture API + Celery logs for the `test_full.py` run around exam/study-plan calls to determine whether prerequisites are unmet by design or whether topic discovery/study-plan gating is incorrect for this content type.
3. Optionally run `test_full.py` again after (1) to separate Gemini 404 from quota behavior.


---

## 13) Live rerun summary (2026-11-21, after fixes)
- **Gemini model change:** switched `GOOGLE_VISION_MODEL` default/config + `.env` and test scripts to `gemini-flash-latest` (replacing unavailable `gemini-2.5-flash`). No secrets written to reports.
- **`test_models_report.py` (ml env):** rerun generated `MODEL_TEST_REPORT.md` (`2026-11-21T15:25:08.336374Z`). PASS: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `all-MiniLM-L6-v2`, `provider-lock (Groq-only)`. FAIL: `gemini-flash-latest` with **ServerError 503 UNAVAILABLE** (“high demand”, transient/external).
- **`test_full.py` (API http://127.0.0.1:8002/api):** rerun generated `FULL_TEST_REPORT.md` (`2026-11-21T15:35:19.349562Z`, run tag `20261121153407`). **Total 24, Passed 22, Failed 2, Info 0.** Key passes include upload→index, RAG chat (ar/en), exam generation/submission + answer key hidden, study plan, dashboards/teacher/negative auth checks. Failures: (1) Gemini image model smoke **503 UNAVAILABLE** (`gemini-flash-latest`); (2) `POST /student/analyze-image` **HTTP 429** (Gemini quota/rate limit). These are **external/service/quota** conditions.
- **Code fix (root cause addressed):** improved fallback topic extraction in `educore/services/student_service.py` so Arabic TXT content used by `test_full.py` yields discoverable topics when LLM topic JSON is absent/empty; this resolved prior “No topics discovered” and the downstream study-plan 404 in the latest full rerun.
