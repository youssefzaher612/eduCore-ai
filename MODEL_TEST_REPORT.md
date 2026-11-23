# EduCore Model Testing Report

Generated: 2026-11-21T15:28:17.272653Z

| Model | Type | Status | Latency (ms) | Error | Sample |
|-------|------|--------|--------------|-------|--------|
| openai/gpt-oss-120b | llm | PASS | 1324.9 |  |  |
| openai/gpt-oss-20b | llm | PASS | 604.33 |  |  |
| all-MiniLM-L6-v2 | embedding | PASS | 5116.71 |  | 384-dim vectors for 3 texts |
| gemini-flash-latest | image | FAIL | - | ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}} |  |
| provider-lock (Groq-only) | config | PASS | - |  | Text generation locked to Groq provider |
