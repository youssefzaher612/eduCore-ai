# AI Usage Monitoring

## Existing signals

- `rag/generator.py` logs response character counts, model, and total token usage when provided by the Groq-compatible API. Streaming calls do not retain provider usage in the current implementation.
- `teacher_ai/core/llm_client.py` normalizes prompt/completion/total tokens, latency, model, request ID, and attempt count into `LLMResponse`; normal success logs include completion token count and latency.
- Gemini image calls do not write a normalized usage record.
- No persistent usage table, per-feature counters, dashboard, cost estimator, cache hit rate, provider health history, or quota enforcement currently exists.

## Reporting requirement

Record provider, model, feature, role, timestamp, latency, success/error class, retry count, provider-reported input/output tokens, and cache result. Use an opaque internal user identifier only when needed for authorized usage aggregation; never record prompts, document text, API keys, or student performance payloads. Cost estimates require provider pricing configuration and should be labeled estimates, not invoices.

## Admin view target

Provide total requests, requests by provider/feature, reported input/output tokens, estimated cost, cache hit rate, latency, failures, rate-limit events, and provider availability. Role-separated summaries require authenticated admin authorization and should exclude identifying or sensitive payloads.
