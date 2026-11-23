# Demo Runbook

## Before the demo

1. From the repository root, install `.[dev,streamlit]` and configure `.env` with PostgreSQL, Celery broker, and the desired provider key(s).
2. Verify PostgreSQL with pgvector and the broker are running.
3. Start the API, Celery worker, and Streamlit frontend using the commands in `AI_PROVIDER_SETUP.md`.
4. Visit `/api/health` and `/docs`; verify an indexed document is available before demonstrating RAG.
5. For a minimal external-spend demo, keep web search disabled, use local embeddings, and avoid repeated provider calls; current response caching and provider quotas are not implemented.

## Demo scenarios

- Teacher: upload course material, wait for indexing, ask a source-grounded question, generate a lesson or assessment, inspect citations, and review authorized assessment records.
- Student: ask a course question, generate practice questions, submit answers and verify deterministic grading, create a study plan, and request analysis based on own recorded attempts.
- Vision: use a small image and verify `GOOGLE_API_KEY` is configured.

## Operational limits

This package has no public admin monitoring dashboard, persistent AI usage ledger, semantic response cache, or authenticated principal integration visible in routes. Do not use the current API as a public multi-tenant deployment until route identity is bound to trusted authentication. The demo mode setting is currently configuration only and does not enforce quotas.
