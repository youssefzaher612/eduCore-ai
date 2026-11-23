# Local model inventory

| Feature | Model | Size / vector shape | Notes |
|---|---|---:|---|
| Chat, RAG, teacher tools, study plans, exams | `qwen3-vl:2b-instruct-q4_K_M` | 1.9 GB, Q4_K_M | Local Ollama text + image model, shared with vision so only one model download is needed. |
| Image questions and diagrams | `qwen3-vl:2b-instruct-q4_K_M` | 1.9 GB, Q4_K_M | Ollama model supports text and image input; image payloads use the local OpenAI-compatible endpoint. |
| Document and query embeddings | `all-MiniLM-L6-v2` | 384 dimensions | Existing local Sentence Transformers model. Retained to preserve compatibility with the current pgvector column and existing vectors. |

## Hardware fit

The inspected machine is Windows 11 Pro, AMD Ryzen 5 6600H (6 cores / 12 threads), 16 GB system RAM, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), and about 140 GB free on `D:`. Only about 2.2 GB system RAM was free during inspection. The 2B quantized multimodal model is the practical choice; larger 4B vision or 7B text models would compete with the current desktop workload and 4 GB GPU memory. Ollama is configured for a 4096 token context, and its model store is on `D:`.

The embedding model remains separate from Ollama because it is already local, returns the exact 384 dimensions the database expects, and does not incur a paid API call. Replacing it with a different embedding model—even one with the same dimension—would change vector space and require re-indexing every document.

## Download and verify

```powershell
ollama pull qwen3-vl:2b-instruct-q4_K_M
ollama show qwen3-vl:2b-instruct-q4_K_M
ollama list
```
