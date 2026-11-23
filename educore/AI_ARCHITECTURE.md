# EduCore AI Architecture

## Local-first request flow

```text
FastAPI route
  -> existing teacher/student service and authorization checks
     -> PostgreSQL/pgvector reads scoped to project and optional document
     -> local Sentence Transformers embeddings (all-MiniLM-L6-v2, 384D)
     -> Ollama OpenAI-compatible chat API for text and image requests
     -> output/schema and citation validation
     -> existing API response and streaming contract
```

`AI_PROVIDER=ollama` is the default. The existing Groq and OpenRouter adapters remain optional compatibility providers. Student image analysis uses the configured local Ollama vision model in local mode; Gemini is optional when a remote provider is explicitly selected. Teacher specialist tools use the shared `create_sync_client` provider adapter, while chat and student AI use `create_async_client`.

## Models and resource limits

The demo model is `qwen3-vl:2b-instruct-q4_K_M` for both text and images. Sharing one small multimodal model limits downloads and avoids keeping separate text and vision models in memory. `OLLAMA_NUM_CTX` defaults to 4096 to fit this machine's memory, and each generation keeps the model resident for 30 minutes to avoid repeated cold starts. Student exam token ceilings scale with the requested question count, so one-question practice uses a much smaller cap than a full exam. The existing Sentence Transformers embedding model remains local and stays at 384 dimensions, so current pgvector data does not need destructive migration or re-indexing.

Text requests use the Ollama OpenAI-compatible `/v1/chat/completions` API. Image requests pass validated, compressed image bytes as a data URL to the same multimodal model. Ingestion, extraction, chunking, and vector retrieval remain local. Extraction results are cached in process by project/document scope, file type, and SHA-256 file content version. Embeddings are cached by model and text hash. Exact and near-duplicate RAG answers use short-lived process caches scoped by user, project, role, task, model, retrieved excerpts, and conversation context. Semantic reuse additionally requires a cosine similarity of at least 0.965 and identical source context.

## Data and authorization boundaries

Retrieval continues to filter by project, indexed document status, and optional document ID. Teacher/student permission checks and deterministic scoring remain in backend services. The current app does not bind request IDs to a verified authenticated principal; it should remain a local demonstration and must not be exposed as a public multi-tenant service until authentication and role-bound routes are enforced. The response cache is not a substitute for authentication and is never shared across distinct user/project/role scopes.

Do not cache vision answers or personal performance analyses. Chat cache entries expire after five minutes and are invalidated naturally when prompt, history, model, role, user, project, or retrieved source text changes. The embedding cache stores SHA-256 keyed local vectors, not raw request text. All caches are in process, bounded, and cleared on restart.

## Optional remote providers

Set `AI_PROVIDER=groq` or `AI_PROVIDER=openrouter` only when remote inference is intended. Their API keys remain optional configuration. Google Gemini remains a separate optional image provider for non-Ollama deployments. Local mode does not read or require Google, Groq, OpenRouter, or OpenAI credentials.
