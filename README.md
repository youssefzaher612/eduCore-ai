# 🎓 EduCore — RAG-Powered Educational Assistant

A complete, production-ready **RAG (Retrieval-Augmented Generation)** system for education, with separate **Student** and **Teacher** sections.

Built with **FastAPI**, **PostgreSQL + pgvector**, **Groq** for text generation, **Celery + RabbitMQ**, and **Streamlit**.

---

## ✨ Features

- 📄 **Document Upload** — PDF, TXT, Markdown, CSV, XLSX
- 🔍 **Semantic Search** — pgvector cosine similarity with HNSW index
- 🤖 **RAG Q&A** — Answers grounded in your uploaded documents
- 📌 **Citations** — Page numbers and source references
- 💬 **Chat History** — Conversational context across messages
- ⚡ **Streaming** — Real-time SSE response streaming
- 👨‍🎓 **Student Section** — Simple explanations with practical examples
- 👨‍🏫 **Teacher Workspace** — RAG-grounded lesson preparation, assessment generation, grading, student analytics, reports and communication
- 🔄 **Background Processing** — Celery + RabbitMQ for async ingestion

---

## 🏗️ Architecture

```
Streamlit UI ──→ FastAPI API ──→ RAG Engine ──→ Groq (LLM)
                      │                │
                      │                ├──→ sentence-transformers (Embeddings)
                      │                └──→ pgvector (Vector Search)
                      │
                      └──→ Celery Workers ──→ RabbitMQ
                                │
                                └──→ Redis (Results)
```

---

## 🚀 Quick Start

### 1. Start infrastructure

```bash
cd EduCore
docker compose up -d
```

This starts PostgreSQL+pgvector, RabbitMQ, and Redis.

### 2. Install dependencies

```bash
pip install -e ".[streamlit]"
# or with uv:
uv sync --extra streamlit
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY (create one at https://console.groq.com/keys)
```

### 3.1 Enable OCR for scanned PDFs (Windows)

Scanned PDFs contain page images instead of selectable text. Install Tesseract and the English/Arabic language data once:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ocr.ps1
```

The worker automatically uses OCR when a PDF page has no text layer. Docker images install Tesseract and both language packs during build.

### 4. Run database migrations

```bash
alembic upgrade head
```

### 5. Start the application

```bash
# Terminal 1: FastAPI server
uvicorn educore.main:app --reload

# Terminal 2: Celery worker
celery -A educore.celery_app worker --loglevel=info -Q educore.process

# Terminal 3: Streamlit UI
streamlit run streamlit_app.py
```

### 6. Open the UI

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **RabbitMQ Dashboard**: http://localhost:15672 (guest/guest)

---

## 📁 Project Structure

```
EduCore/
├── educore/                    # Main Python package
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Pydantic Settings (env vars)
│   ├── celery_app.py           # Celery configuration
│   ├── database/
│   │   ├── connection.py       # Async + sync DB sessions
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   └── repositories.py     # Data access layer
│   ├── rag/
│   │   ├── extractor.py        # PDF/TXT/CSV/XLSX text extraction
│   │   ├── chunker.py          # Text splitting + undersized merge
│   │   ├── embedder.py         # sentence-transformers embeddings
│   │   ├── retriever.py        # pgvector semantic search
│   │   ├── generator.py        # OpenRouter LLM streaming generation
│   │   └── prompts.py          # System prompts (student/teacher)
│   ├── services/
│   │   ├── ingestion_service.py  # Upload orchestration
│   │   ├── chat_service.py       # RAG Q&A pipeline
│   │   └── teacher_service.py    # Teacher course retrieval and real exam records
│   ├── teacher_ai/               # Embedded teacher modules and lesson verification
│   ├── tasks/
│   │   └── ingestion_tasks.py  # Celery background tasks
│   └── api/
│       ├── deps.py             # FastAPI dependency injection
│       └── routes.py           # All API endpoints
├── streamlit_app.py            # Streamlit frontend
├── docker-compose.yml          # PostgreSQL, RabbitMQ, Redis
├── Dockerfile                  # Multi-stage build
├── alembic.ini                 # Migration config
├── alembic/
│   ├── env.py                  # Migration environment
│   └── versions/
│       └── 001_initial_schema.py
├── pyproject.toml              # Dependencies & build config
├── .env.example                # Configuration template
└── storage/                    # Uploaded files (git-ignored)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/users` | Create user |
| `GET` | `/api/users` | List users |
| `POST` | `/api/projects` | Create project |
| `GET` | `/api/projects` | List projects |
| `POST` | `/api/documents/upload` | Upload document |
| `GET` | `/api/documents` | List documents |
| `GET` | `/api/documents/{id}/status` | Document processing status |
| `POST` | `/api/chat/sessions` | Create chat session |
| `GET` | `/api/chat/sessions` | List chat sessions |
| `GET` | `/api/chat/sessions/{id}/messages` | Get chat messages |
| `POST` | `/api/chat/sessions/{id}/ask` | Ask a question (SSE; `mode`: `study` or `hint`) |
| `GET` | `/api/student/dashboard` | Student accuracy, attempts, and BKT mastery by topic |
| `POST` | `/api/student/practice-question` | Generate an adaptive practice question from indexed course material |
| `POST` | `/api/student/study-plan` | Build a weekly plan based on topic mastery |
| `POST` | `/api/student/exams` | Generate a quiz from indexed course documents |
| `POST` | `/api/student/exams/{id}/submit` | Grade an exam and save its attempts |
| `POST` | `/api/student/analyze-image` | Answer a question about a temporary image upload |
| `GET` | `/api/teacher/materials` | Retrieve topic excerpts from one selected indexed course file |
| `GET` | `/api/teacher/assessment-records` | Read submitted EduCore exam results for teacher analytics |

---

## ⚙️ Configuration

All settings are in `.env` — see `.env.example` for the full list:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `groq` | Text generation uses Groq |
| `GROQ_API_KEY` | — | Groq API key when `LLM_PROVIDER=groq` |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible endpoint |
| `GROQ_CHAT_MODEL` | `openai/gpt-oss-120b` | Student and teacher text generation on Groq |
| `GROQ_FAST_MODEL` | `openai/gpt-oss-20b` | Fallback generation model on Groq |
| `GROQ_WEB_SEARCH_ENABLED` | `false` | Enable Groq built-in browser search only when requested |
| `OPENROUTER_API_KEY` | — | Legacy setting; not used for text or image generation |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Legacy setting; text generation uses `GROQ_BASE_URL` |
| `GOOGLE_API_KEY` | — | Google AI Studio key for image understanding |
| `GOOGLE_VISION_MODEL` | `gemini-2.5-flash` | Gemini image understanding model |
| `OPENROUTER_ENABLE_WEB_SEARCH` | `false` | Enable separately billed external web search only when desired |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `EMBEDDING_DIMENSIONS` | `384` | Must match the embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `RETRIEVAL_MIN_SCORE` | `0.3` | Minimum cosine similarity |
| `CHAT_HISTORY_LIMIT` | `10` | Messages in chat context |

---

## 🧠 RAG Pipeline

```
Document Upload
    │
    ▼
Text Extraction (PyMuPDF / plain text / CSV / XLSX)
    │
    ▼
Text Sanitization (NFKC normalize, strip NUL, collapse whitespace)
    │
    ▼
Chunking (RecursiveCharacterTextSplitter + undersized merge)
    │
    ▼
Embedding (sentence-transformers, local, free)
    │
    ▼
Vector Storage (pgvector HNSW index)
    │
    ▼
Question → LLM rewrite for lecture search → Embed → Cosine Search → Top-K Chunks
    │
    ▼
Prompt Assembly (system prompt + context + history + question)
    │
    ▼
Streaming Generation (Groq API → SSE)
```

---

## 🎓 Student AI integration

The student interface now includes progress, adaptive practice, and exam tabs alongside the existing RAG chat and course library. Student and teacher text-generation features share EduCore's Groq configuration and PostgreSQL database.

- Practice answers are graded on the server and saved by student and course.
- Topic progress combines observed accuracy with Bayesian Knowledge Tracing mastery estimates.
- The study plan prioritizes the weakest topics after at least three attempts; before then, it balances practice across topics.
- The course topic list is extracted from the selected course's indexed RAG files, so a student's subject and topics follow that teacher's materials rather than the notebook's English topic list.
- Adaptive practice, exams, mastery tracking, and study plans use only the selected course's indexed files. Without indexed files, topic choices stay empty until the teacher's material is indexed.
- Students can also ask about a temporary image; it is not added to the shared course library.
- Correct answers remain server-side until an exam is submitted. Exam answers are then saved as learning attempts.
- Student questions about their own progress, weak topics, or study plan are answered from recorded attempts in the existing chat. Course-content questions continue through the RAG evidence flow.
- The student chat has two response modes: **Study** explains and teaches; **Hint** gives a nudge without revealing the final answer or correct option.
- Questions in Arabic, English, or dialect are rewritten into concise bilingual search terms before lecture retrieval. Answers stay in the user's language and style.
- Uploaded lectures are the default source. Web search is disabled by default; when explicitly enabled, unsupported answers ask for consent before searching, while an explicit web request counts as consent. External answers combine browser search with model synthesis and show source links when available.
- External web search uses the configured provider's browser tool; availability and charges depend on the selected provider and model.

Apply the database migrations before using these features:

```bash
alembic upgrade head
```

Generated assessments still need teacher review before classroom use.

---

## 📝 License

MIT
