# EduCore Full Testing Report

Generated: 2026-11-21T15:35:19.349562Z

Total: 24 | Passed: 22 | Failed: 2 | Info: 0

| Test | Status | Latency (ms) | Detail |
|------|--------|--------------|--------|
| Groq primary LLM | PASS | 1135.1 | Answer:  |
| Groq fallback LLM | PASS | 668.69 | Answer:  |
| Embedding model | PASS | 8080.76 | 384-dim, 2 texts, encode=39ms |
| Gemini image model | FAIL | - | ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is  |
| Document upload | PASS | 18.12 | doc_id=b7f60472-8953-4b12-a1ec-d5085bdac323, status=pending |
| Document indexing | PASS | - | chunks=2 |
| RAG chat [ar/study] | PASS | 2960.97 | answer_len=288, citations=1 |
| RAG chat [en/study] | PASS | 2145.66 | answer_len=617, citations=1 |
| RAG chat [ar/study] | PASS | 3350.17 | answer_len=614, citations=1 |
| RAG chat [en/hint] | PASS | 3684.28 | answer_len=276, citations=1 |
| Document search | PASS | - | excerpts=2, doc=test_lecture.txt |
| Web search (disabled) | PASS | 26.02 | response_len=154, disabled_msg_present=True |
| Exam generation | PASS | - | topic=مقدمة في الذكاء الاصطناعي, exam_id=1667a66f-247a-4f9c-a7f6-460fbeb25bc7, q |
| Exam answer key hidden | PASS | - | 0/3 questions expose an answer key |
| Exam submission | PASS | - | score=66.7%, correct=2/3, review_consistent=True |
| Study plan | PASS | - | topic=مقدمة في الذكاء الاصطناعي, hours=4, plan_days=7, summary=خطة أسبوعية لتقوي |
| Student dashboard | PASS | - | attempts=3, topics=12, course_topics=12 |
| Teacher assessment records | PASS | - | records=1 |
| Image analysis | FAIL | 871.82 | HTTP 429: {"detail":"Gemini quota or request limit reached. Check Google AI Stud |
| Wrong user (dashboard) | PASS | - | HTTP 404: {"detail":"Student or course was not found"} |
| Wrong user (teacher records) | PASS | - | HTTP 400: {"detail":"الحساب أو المقرر مش موجود."} |
| Wrong user (delete document) | PASS | - | HTTP 409: {"detail":"Not authorized to delete this document"} |
| Wrong user (upload document) | PASS | - | HTTP 400: {"detail":"Not authorized to access this project"} |
| Delete document | PASS | - | status_after_delete=HTTP 404 |

## Evaluation Entries (RAG Chat)

| Question | Mode | Answer Length | Citations | Arabic |
|----------|------|---------------|-----------|--------|
| ما هو المعالجة اللغوية الطبيعية؟ اشرحها ببساطة. | study | 288 | 1 | True |
| What is NLP and what are its key applications? | study | 617 | 1 | False |
| كمل الشرح عن تطبيقات NLP. | study | 614 | 1 | True |
| Explain machine learning algorithms briefly. | hint | 276 | 1 | False |

## Environment

- Python: `3.10.20`
- API: `http://127.0.0.1:8002/api`
- Run tag: `20261121153407`
- Chat model: `openai/gpt-oss-120b`
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
- Vision model: `gemini-flash-latest`
- Web search flags: OPENROUTER_ENABLE_WEB_SEARCH=`false`, GROQ_WEB_SEARCH_ENABLED=`false`

## Notes on test design

- Exam questions returned by the API intentionally omit `correct_index`; submission is validated against the score and review returned by the server.
- Web search is validated in its disabled-by-configuration state; no live external provider calls are made.
- Arabic content is uploaded as TXT due to the known Arabic PDF extraction defect.

## Residual risks

- Topic discovery and exam generation depend on LLM structured output and can hit provider rate limits.
- Arabic PDF extraction remains unverified/unfixed.
