# EduCore local AI setup

EduCore defaults to local Ollama inference. The chosen model is a small Qwen3-VL multimodal model so chat and image analysis share one resident model. The existing `all-MiniLM-L6-v2` Sentence Transformers embedding model stays local at 384 dimensions; keeping its vector space avoids changing or rebuilding existing pgvector rows.

## Install and start Ollama

Ollama is installed under `D:\Ollama`; model blobs are stored under `D:\Ollama\models` so the full `C:` drive is not used for model downloads. The installer sets the user-level `OLLAMA_MODELS` variable.

```powershell
ollama --version
ollama list
```

If Ollama is not already running from the Windows app, start it in a separate terminal:

```powershell
ollama serve
```

Download and inspect the configured local model:

```powershell
ollama pull qwen3-vl:2b-instruct-q4_K_M
ollama list
curl.exe http://localhost:11434/api/tags
```

## Configure EduCore

The repository `.env.example` uses `AI_PROVIDER=ollama` and defaults to:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3-vl:2b-instruct-q4_K_M
OLLAMA_VISION_MODEL=qwen3-vl:2b-instruct-q4_K_M
OLLAMA_NUM_CTX=4096
OLLAMA_KEEP_ALIVE=30m
OLLAMA_TIMEOUT_SECONDS=180
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
```

No Google, Groq, OpenAI, or OpenRouter credential is required in local mode. `LLM_PROVIDER=groq` and `LLM_PROVIDER=openrouter` remain available for an intentional remote deployment.

## Start the application

From the repository root (the directory containing `pyproject.toml` and `streamlit_app.py`):

```powershell
python -m pip install -e ".[dev,streamlit]"
uvicorn educore.main:app --host 127.0.0.1 --port 8002
```

In a second terminal at the same repository root:

```powershell
streamlit run streamlit_app.py --server.port 8501
```

Inspect service and model availability:

```powershell
curl.exe http://127.0.0.1:8002/api/health
curl.exe http://127.0.0.1:8002/api/ai/status
```

The `/api/ai/status` response reports whether Ollama answers locally and whether the configured chat model appears in its installed model list. Requests keep the shared model loaded for 30 minutes after the last generation, which avoids repeated cold loads between student and teacher actions. Do not expose Ollama's port `11434` publicly.

## Tests

Run the suite from the repository root:

```powershell
python -m pytest -p no:cacheprovider
```

Local provider unit tests use mocked model responses. Small manual integration prompts should be run with Ollama started and the configured model installed.
