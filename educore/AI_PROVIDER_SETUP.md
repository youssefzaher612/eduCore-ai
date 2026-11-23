# AI Provider Setup

## Current supported configuration

The current code supports Groq-compatible text generation and Google Gemini for student image analysis. Embeddings are generated locally with Sentence Transformers. Provider keys remain backend environment variables.

From the repository root, copy `.env.example` to `.env` and set `GROQ_API_KEY`. Set `GOOGLE_API_KEY` only if image analysis is needed. Keep `.env` out of version control. The repository currently has a default local PostgreSQL connection and Celery broker settings; configure these for your environment.

Relevant settings include `LLM_PROVIDER=groq`, `GROQ_BASE_URL`, `GROQ_CHAT_MODEL`, `GROQ_FAST_MODEL`, `GROQ_WEB_SEARCH_ENABLED=false`, `GOOGLE_VISION_MODEL`, `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2`, `EMBEDDING_DIMENSIONS=384`, `OPENROUTER_MAX_TOKENS=8192` (hard ceiling; normal chat defaults to 2,048), `AI_MAX_OUTPUT_TOKENS=8192`, `RETRIEVAL_TOP_K=3`, `RETRIEVAL_MIN_SCORE=0.3`, `RETRIEVAL_MAX_CONTEXT_CHARS=9000`, `CHAT_HISTORY_LIMIT=6`, `CHAT_HISTORY_MESSAGE_CHARS=2000`, and `DEMO_MODE=false`.

## Dependencies and commands

Run from the repository root (the parent of `educore/`):

```powershell
python -m pip install -e ".[dev,streamlit]"
Copy-Item .env.example .env
uvicorn educore.main:app --reload
celery -A educore.celery_app.celery_app worker --loglevel=info
streamlit run app.py
```

Configure and start PostgreSQL with pgvector, plus the configured Celery broker/result backend, before running ingestion. The exact commands to start those services depend on the local installation and are not checked into this backend package.

## Provider capability notes

Groq routing is currently text-only in `provider.py`; Gemini is the image provider. Free-tier availability and quotas vary by account and may change. This project does not claim unlimited free usage. Ollama and Gemini text routing are not implemented by the current shared generator.
