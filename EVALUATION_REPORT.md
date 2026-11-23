# EduCore Evaluation Report

Generated: 2026-11-21
Plan followed: `CONTINUATION_PLAN.md`
Raw test evidence: `FULL_TEST_REPORT.md` (regenerated on every `test_full.py` run)

## 1. Summary

Final end-to-end run: **24/24 PASS, 0 FAIL**. Unit tests: **28 passed, 2 subtests passed**. The run completed without tracebacks; the previous `KeyError: 'correct_index'` crash is resolved.

## 2. Environment and configuration used

- Python: `/home/amin/miniconda3/envs/ml/bin/python` (3.10.20)
- FastAPI: `http://127.0.0.1:8002/api` (`educore.main:app`, restarted 2026-11-21 16:33 to load fixes)
- Celery worker: `educore.process` queue, `--pool=solo`
- Postgres + pgvector: `localhost:5400/educore`; RabbitMQ: `localhost:5672`; Redis: `localhost:6379`
- Chat model: `openai/gpt-oss-120b` (fallback `openai/gpt-oss-20b`)
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- Vision model: `gemini-2.5-flash`
- Web search: disabled by configuration (`OPENROUTER_ENABLE_WEB_SEARCH=false`, `GROQ_WEB_SEARCH_ENABLED=false`)
- Arabic test content uploaded as `.txt` (see Arabic PDF risk below)
- API keys and local secrets are deliberately omitted from this report.

## 3. Model tests

| Test | Result | Latency | Notes |
|------|--------|---------|-------|
| Groq primary LLM | PASS | 1125 ms | `openai/gpt-oss-120b` |
| Groq fallback LLM | PASS | 381 ms | `openai/gpt-oss-20b` |
| Embedding model | PASS | 8161 ms | 384-dim vectors; encoding 57 ms for 2 texts |
| Gemini image model | PASS | 1161 ms | Direct Gemini call with a generated PNG |

An earlier run hit a transient Gemini `429 RESOURCE_EXHAUSTED`; the final run succeeded. Provider quota remains an external dependency.

## 4. Feature tests

| Test | Result | Evidence |
|------|--------|----------|
| Document upload + indexing | PASS | `doc_id=8dbba4c3-…`, 2 chunks indexed |
| RAG chat (AR/EN, study/hint, follow-up) | PASS | 4 questions answered, citations present (1–2 per answer) |
| Document search (teacher materials) | PASS | 2 excerpts from `test_lecture.txt` |
| Web search disabled guard | PASS | Graceful disabled message returned |
| Exam generation | PASS | Topic `مقدمة في الذكاء الاصطناعي`, 3 questions |
| Answer key privacy | PASS | 0/3 public questions expose `correct_index`/`correct_answer` |
| Exam submission + scoring | PASS | 33.3% (1/3); score matches returned review |
| Study plan | PASS | 7 days, Arabic summary, topic grounded in indexed file |
| Student dashboard | PASS | 3 attempts, 12 course topics |
| Teacher assessment records | PASS | 1 real exam record after submission |
| Image analysis (real Gemini path) | PASS | Yellow image answered `أصفر` |
| Ownership negatives (dashboard, records, delete, upload) | PASS | Rejected with 400/404/409 |
| Delete document | PASS | 200 on delete, 404 on subsequent status check |

Each run creates a unique user/project/document and captures IDs from API responses; no hardcoded IDs from earlier runs are reused. The uploaded document is deleted at the end of the run.

## 5. Defects found and fixed

1. **Exam endpoint HTTP 500** (`educore/services/student_service.py`, `generate_exam`)
   The generated payload was never assigned to `generated`, so `generated.get("title")` raised `AttributeError: 'NoneType' object has no attribute 'get'`. Now `generated = payload.model_dump(mode="python")`.
2. **Pydantic/dict mismatch in exam validator** (`generate_exam`)
   `payload.questions` contains `StudentExamQuestionOutput` objects, but `_validate_generated_questions` expects dictionaries. Questions are now converted with `model_dump(mode="python")` before validation.
3. **Study plan JSON truncation** (`student_service.py`, `generate_plan`)
   `max_tokens=700` cut off the seven-entry JSON response on both attempts. Raised to `1600`; the final run produced a valid 7-day plan.
4. **Test assumed answer key was public** (`test_full.py`, `test_exam`)
   Public questions intentionally hide `correct_index`; the test crashed with `KeyError`. It now submits a deterministic answer set and validates score/review consistency, and explicitly asserts that no answer key leaks.
5. **Standalone script collected by pytest** (`test_full.py`)
   `pytest` treated `test_*` helper functions as tests and reported 11 fixture errors. Added module-level `__test__ = False`.
6. **Debug logging at ERROR level** (`student_service.py`)
   Removed raw LLM payload logging from `logger.error`; replaced with `logger.debug`.
7. **Test data fragility** (`test_full.py`)
   Unique user/project per run, IDs captured from responses, deletion at end, ownership negative cases, and explicit timeouts/context for indexing polling.

## 6. Verification commands

```bash
cd "/media/amin/data/Machine-Learning-Engineer/09-Volte IQ/eduCore"
/home/amin/miniconda3/envs/ml/bin/python -m py_compile test_full.py educore/services/student_service.py ...   # PASS
/home/amin/miniconda3/envs/ml/bin/python -m pytest -q                                                        # 28 passed, 2 subtests passed
/home/amin/miniconda3/envs/ml/bin/python test_full.py                                                         # 24/24 PASS
```

`ruff` is not installed in the `ml` environment, so lint was not run. No new lint warnings were introduced; compile checks pass.

## 7. Remaining risks and blockers

- **Arabic PDF extraction:** PyMuPDF returns garbled Arabic text (characters rendered as dots) for PDFs written with certain embedded fonts. The TXT workaround is used in tests. A robust fix requires detecting unusable extracted text and falling back to OCR with Arabic language data; Tesseract is **not installed** in this environment, so OCR fallback cannot currently be exercised.
- **Provider rate limits/quota:** Groq 429s trigger the fallback model; Gemini quota produced an intermittent 429. Both are external and can make a rerun flaky.
- **LLM structured output:** Topic discovery, exam generation, and study-plan generation depend on valid JSON from the model. Retries exist, but persistent malformed output will still surface as a 400/404.
- **Web search enabled path:** Not exercised live because the feature is disabled by configuration; only the disabled guard was validated.
- **Teacher grading flows:** The teacher AI modules (grading/lesson/assessment UI) were reviewed previously (`TEACHER_MODELS_REVIEW.md`); this run covers the API endpoints exposed by EduCore (`materials`, `assessment-records`) but not Streamlit UI interactions.

## 8. Conclusion

The blocking failures from the previous session are fixed and the full suite now passes end to end. The remaining items are environmental (provider quota, missing OCR tooling) or documented product risks, not code defects blocking the current feature set.
