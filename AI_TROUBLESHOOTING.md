# Local AI troubleshooting

## Ollama is unavailable

Check the service and model list:

```powershell
ollama --version
ollama list
curl.exe http://localhost:11434/api/tags
```

If the Windows Ollama app is not running, start `ollama serve` in a terminal. The API should remain bound to localhost; do not publish port 11434 to the internet.

## The model is missing

```powershell
ollama pull qwen3-vl:2b-instruct-q4_K_M
ollama list
```

Confirm that `OLLAMA_MODELS` points to `D:\Ollama\models`. The selected model requires a recent Ollama release; the official model page lists its current compatibility requirement.

## EduCore uses a remote provider

Check the effective values in the root `.env`:

```dotenv
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3-vl:2b-instruct-q4_K_M
OLLAMA_VISION_MODEL=qwen3-vl:2b-instruct-q4_K_M
```

Restart both backend and frontend after changing environment settings. `AI_PROVIDER` takes precedence; if empty, the older `LLM_PROVIDER` value is used.

## Image requests fail

Use JPG, PNG, or WebP. The backend checks file type and upload size and compresses supported images before sending them to the local vision model. Verify the installed model supports image input and that `/api/ai/status` reports it installed.

## Model is slow or memory is low

The machine has 16 GB RAM and 4 GB GPU memory, with roughly 2.2 GB RAM free at inspection time. Close other memory-heavy programs, keep `OLLAMA_NUM_CTX=4096`, and avoid loading a second model. The app uses one shared 2B text-and-vision model. Model inference runs locally and can be slower than hosted inference on this hardware.

## Retrieval finds no sources

Local inference does not fix an empty or stale vector index. Confirm the document has `INDEXED` status and non-zero chunks, then re-index it if needed. Keep `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2` and `EMBEDDING_DIMENSIONS=384` unless you plan a complete document re-index with a new embedding space.

## API health

```powershell
curl.exe http://127.0.0.1:8002/api/health
curl.exe http://127.0.0.1:8002/api/ai/status
```
