# EduCore AI Optimization Plan

## Scope and current architecture

This checkout is the Python backend package. The parent repository contains the project manifest and tests, but no frontend source is inside this writable package. The backend exposes FastAPI routes and delegates to shared `ChatService`, `StudentLearningService`, `TeacherWorkspaceService`, and teacher AI modules. PostgreSQL/pgvector stores chunks, local Sentence Transformers produces 384 dimensional vectors, and Celery runs document extraction, chunking, embedding, and indexing.

Text generation uses the OpenAI-compatible Groq endpoint through `OpenRouterGenerator` for chat/student functions and the synchronous `LLMClient` for teacher tools. Gemini is called directly for student image analysis. There is no general provider router, Ollama adapter, persisted usage ledger, response cache, or admin monitoring endpoint in this checkout.

## External AI call inventory

Token counts below are request dependent; provider usage is only retained by `LLMClient` responses and mostly logged by `OpenRouterGenerator`. Exact per-feature token values require provider usage telemetry and cannot be reconstructed from current source alone.

| Location / function | Provider and model | Feature | Input/output behavior | Optimization |
|---|---|---|---|---|
| `rag/generator.py:OpenRouterGenerator.generate` | Groq-compatible API, `GROQ_CHAT_MODEL`, fallback `GROQ_FAST_MODEL` | Student practice questions/hints, study plans, query rewriting, non-stream RAG, web search | Input is caller-built messages; output is requested max tokens, default previously inherited 60,000 | Bounded global cap; task-specific output limits; avoid retrying non-rate-limit failures |
| `rag/generator.py:generate_stream` | Same Groq models | Main RAG chat | Input includes system prompt, recent conversation, retrieved passages; default output budget previously 60,000 | Bounded history, retrieval Top-K 3 and context chars cap; add usage capture for non-streaming telemetry |
| `rag/generator.py:generate_with_web_search` | Groq-compatible API | Explicit external search | Input includes recent conversation; output at least 2,048 tokens | Search is already opt-in (`GROQ_WEB_SEARCH_ENABLED=false`); preserve explicit opt-in and cap request |
| `teacher_ai/core/llm_client.py:LLMClient._call_provider` via `generate_detailed` | Groq, configured model then fast model | Lesson preparation/review/repair, assessment generation, grading, reports, communication, analytics insight | Bounded at 4,096 output tokens; response usage and latency available in `LLMResponse` | Consolidate duplicated provider logic and track metrics; reduce oversized per-task limits where schemas allow |
| `services/student_service.py:analyze_image` | Google Gemini, `GOOGLE_VISION_MODEL` | Student uploaded image/question analysis | Image plus prompt and up to six history messages | Keep vision-capability routing; validate/compress/deduplicate uploads and avoid forwarding unnecessary history |
| `rag/embedder.py:Embedder.embed_texts` | Local Sentence Transformers, `EMBEDDING_MODEL_NAME` | Ingestion embeddings | No provider token cost; model runs on CPU | Existing vectors are reused by retrieval; skip re-embedding byte-identical uploads and cache repeated query vectors |

### Not an external call

`services/chat_service.py` may call `rag/query_rewriter.py` before retrieval; this is a Groq generation call and is included in the first inventory row. `StudentLearningService` calculates exam grading, performance metrics, progress, and schedule dates in Python/database logic; these do not require an LLM for numerical correctness.

## Cost and reliability findings

- `OPENROUTER_MAX_TOKENS` defaulted to 60,000 despite the app’s feature requests generally needing hundreds to a few thousand tokens.
- `RETRIEVAL_TOP_K=5`; chat history is configured at 10 messages and prompt construction allowed ten messages of up to 6,000 characters each.
- The chat query rewrite is an extra generation before answer generation in contextual cases. It should remain conditional and be measured before broadening its use.
- Groq generation has a second model fallback, and teacher calls can retry twice. `OpenRouterGenerator` sleeps for 2 and 4 seconds on rate limits, which may create unnecessary latency; provider cooldown/circuit-breaker behavior remains to be implemented.
- Embeddings are local and dimension-checked. There is no persisted extraction hash or embedding cache for document re-indexing.
- Exact/semantic answer caching is not safe to add without a trustworthy authenticated principal and complete source/version/permission scope. The current API accepts user IDs in request payloads and no auth dependency is visible in `api/deps.py`; therefore cache answers must not be shared and private analytics authorization remains unverified.
- No demo usage quotas, persistent monitoring dashboard, teacher/student provider router, provider health checks, local LLM fallback, or answer-cache invalidation exists.

## Security and permission boundary

The service layer does validate project ownership for several teacher/student operations and project-scoped chunk retrieval is enforced in SQL. Student analytics are constrained to the selected student in service logic. However, the API routes shown accept `user_id`/`project_id` in request bodies and the API dependency layer has no authenticated-principal dependency. Caller-controlled IDs are not authentication. Before a public deployment, bind user identity to a verified auth principal and enforce role checks at each route; do not use frontend filtering as authorization. Do not log document contents or raw student records.

## Changes completed in this pass

- Lowered the default shared generation output budget to 2,048 tokens, with configurable explicit task budgets up to a hard ceiling of 8,192 tokens.
- Set default RAG Top-K to 3 while retaining a configurable 1–5 range.
- Bounded assembled RAG source context to 9,000 characters.
- Bounded chat history to six messages and 2,000 characters per message by default.
- Added a bounded in-process LRU for identical local query embeddings (2,048 entries), keyed by a SHA-256 digest so raw queries are not retained as cache keys. It is model-keyed and does not mix vector dimensions.
- Added configuration placeholders for demo mode, request input/output limits, and provider cooldown thresholds. These do not claim to be enforced yet.

## Implementation checklist

- [x] Phase 1 audit and call inventory (source-level; dynamic telemetry still needed).
- [x] First safe token/history/RAG bounds.
- [ ] Shared provider router with capability-aware routing and one retry/circuit-breaker policy.
- [ ] Usage tracking and per-feature quotas with provider reported token values.
- [ ] Authenticated principal and role-bound endpoint enforcement.
- [ ] File hash/extracted-text reuse and safe document version invalidation.
- [ ] Exact/semantic response cache with user, role, source version, task, prompt, and model scope.
- [ ] Monitoring/admin usage endpoints and UI integration.
- [ ] Demo runbook verification against a running database, worker, and frontend.
- [ ] Teacher/student regression suite and mocked provider integration tests.

## Known limitation

No feature is marked fully complete from this pass. The package was inspected statically; runtime/database availability and the complete frontend are outside this checkout. Exact savings and actual token distributions require instrumented provider usage after deployment.
