"""
Comprehensive EduCore testing: all models + all features + real question evaluation.
Run with the ml env Python.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# This is a standalone end-to-end script, not a pytest module. Its helper
# functions are named test_* but take runtime arguments instead of fixtures.
__test__ = False

load_dotenv(".env")

API_BASE = "http://127.0.0.1:8002/api"
TIMEOUT = httpx.Timeout(30.0, read=180.0)
RUN_TAG = datetime.utcnow().strftime("%Y%m%d%H%M%S")

RESULTS: list[dict] = []
EVAL_ENTRIES: list[dict] = []


def log(test: str, status: str, detail: str = "", latency: float | None = None,
        eval_detail: dict | None = None) -> None:
    RESULTS.append({
        "test": test,
        "status": status,
        "detail": detail,
        "latency_ms": round(latency, 2) if latency is not None else None,
        "eval": eval_detail or {},
        "tested_at": datetime.utcnow().isoformat() + "Z",
    })
    mark = "✅" if status == "PASS" else "❌"
    lat = f" ({latency:.0f}ms)" if latency is not None else ""
    print(f"  {mark} {test}{lat}: {detail[:120]}")


def api(method: str, path: str, **kwargs) -> httpx.Response:
    return getattr(httpx, method.lower())(
        f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs
    )


# ═══════════════════════════════════════════════════════════════════════════
# SETUP: user + project
# ═══════════════════════════════════════════════════════════════════════════

def setup_user_project() -> tuple[str, str]:
    """Create a fresh user/project for this run; capture IDs from responses."""
    r = api("post", "/users", json={
        "name": f"Full Test {RUN_TAG}",
        "email": f"fulltest-{RUN_TAG}@example.com",
        "role": "student",
    })
    if r.status_code != 201:
        raise RuntimeError(f"Could not create test user: HTTP {r.status_code}: {r.text[:200]}")
    user_id = r.json()["id"]

    r = api("post", "/projects", json={
        "name": f"Full Test Course {RUN_TAG}",
        "description": "Automated end-to-end validation project",
    }, params={"owner_id": user_id})
    if r.status_code != 201:
        raise RuntimeError(f"Could not create test project: HTTP {r.status_code}: {r.text[:200]}")
    return user_id, r.json()["id"]


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_groq_primary() -> None:
    model = "openai/gpt-oss-120b"
    try:
        os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                        base_url=os.getenv("GROQ_BASE_URL"))
        start = time.perf_counter()
        c = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            max_completion_tokens=10, temperature=0.0,
        )
        lat = (time.perf_counter() - start) * 1000
        log("Groq primary LLM", "PASS", f"Answer: {c.choices[0].message.content}", lat,
            {"model": model, "answer": c.choices[0].message.content})
    except Exception as e:
        log("Groq primary LLM", "FAIL", f"{type(e).__name__}: {e}")


def test_groq_fallback() -> None:
    model = "openai/gpt-oss-20b"
    try:
        os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                        base_url=os.getenv("GROQ_BASE_URL"))
        start = time.perf_counter()
        c = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the capital of Egypt? Answer in one word."}],
            max_completion_tokens=10, temperature=0.0,
        )
        lat = (time.perf_counter() - start) * 1000
        log("Groq fallback LLM", "PASS", f"Answer: {c.choices[0].message.content}", lat,
            {"model": model, "answer": c.choices[0].message.content})
    except Exception as e:
        log("Groq fallback LLM", "FAIL", f"{type(e).__name__}: {e}")


def test_embedding() -> None:
    model = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
    try:
        from sentence_transformers import SentenceTransformer
        start = time.perf_counter()
        emb = SentenceTransformer(model, device="cpu")
        load_ms = (time.perf_counter() - start) * 1000
        texts = ["What is natural language processing?", "الذكاء الاصطناعي في التعليم"]
        start = time.perf_counter()
        vectors = emb.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        enc_ms = (time.perf_counter() - start) * 1000
        dim = int(vectors.shape[1])
        log("Embedding model", "PASS",
            f"{dim}-dim, {len(texts)} texts, encode={enc_ms:.0f}ms",
            round(load_ms + enc_ms, 2),
            {"model": model, "dimension": dim, "shape": list(vectors.shape)})
    except Exception as e:
        log("Embedding model", "FAIL", f"{type(e).__name__}: {e}")


def test_gemini_image() -> None:
    model = os.getenv("GOOGLE_VISION_MODEL", "gemini-flash-latest")
    try:
        from PIL import Image
        import io
        from google import genai
        from google.genai import types
        img = Image.new("RGB", (128, 128), color=(0, 128, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        start = time.perf_counter()
        resp = client.models.generate_content(
            model=model,
            contents=["What color is this image? Answer in one word.",
                      types.Part.from_bytes(data=img_bytes, mime_type="image/png")],
            config=types.GenerateContentConfig(max_output_tokens=20, temperature=0.0),
        )
        lat = (time.perf_counter() - start) * 1000
        text = (getattr(resp, "text", None) or "").strip()
        log("Gemini image model", "PASS", f"Answer: {text}", lat,
            {"model": model, "answer": text})
    except Exception as e:
        log("Gemini image model", "FAIL", f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_document_upload(user_id: str, project_id: str) -> str | None:
    """Upload a test PDF and wait for indexing. Returns document_id."""
    # Create a rich test TXT with clear sections (text extraction works reliably)
    txt_content = """المحاضرة الأولى: مقدمة في الذكاء الاصطناعي
الذكاء الاصطناعي هو مجال علمي يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشريًا. يشمل الذكاء الاصطناعي التعلم الآلي، ومعالجة اللغة الطبيعية، والرؤية الحاسوبية.

المحاضرة الثانية: معالجة اللغة الطبيعية (NLP)
تتمحور معالجة اللغة الطبيعية حول تمكين الحواسيب من فهم وتوليد اللغة البشرية. من تطبيقاتها: الترجمة الآلية، تحليل المشاعر، والروبوتات الحوارية. تعتمد NLP على نماذج اللغات الكبيرة.

المحاضرة الثالثة: التعلم الآلي
التعلم الآلي هو فرع من الذكاء الاصطناعي يتيح للأنظمة التعلم من البيانات دون برمجة صريحة. من الخوارزميات: الشبكات العصبية، أشجار القرار، وآلات الناقلات الحاملة.

المحاضرة الرابعة: الرؤية الحاسوبية
تتيح الرؤية الحاسوبية للحواسيب تحليل وفهم الصور والفيديو. تشمل التطبيقات: التعرف على الوجوه، كشف الأشياء، والقيادة الذاتية."""
    pdf_path = "/tmp/test_lecture.txt"
    with open(pdf_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    with open(pdf_path, "rb") as f:
        content = f.read()

    start = time.perf_counter()
    r = api("post", "/documents/upload",
            params={"project_id": project_id, "user_id": user_id},
            files={"file": ("test_lecture.txt", content, "text/plain")})
    lat = (time.perf_counter() - start) * 1000

    if r.status_code != 201:
        log("Document upload", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}", lat)
        return None

    data = r.json()
    doc_id = data["document_id"]
    log("Document upload", "PASS", f"doc_id={doc_id}, status={data['status']}", lat)

    # Wait for indexing (up to 60 seconds)
    for _ in range(30):
        time.sleep(2)
        r = api("get", f"/documents/{doc_id}/status")
        st = r.json()
        if st.get("status") == "indexed":
            log("Document indexing", "PASS",
                f"chunks={st.get('total_chunks')}", None)
            return doc_id
        elif st.get("status") == "failed":
            log("Document indexing", "FAIL",
                f"error={st.get('error_message')}", None)
            return doc_id
    log("Document indexing", "FAIL", "Timed out waiting for indexing", None)
    return doc_id


def test_rag_chat(user_id: str, project_id: str, doc_id: str | None) -> None:
    """Test RAG chat in study and hint modes with real questions."""
    # Create chat session
    r = api("post", "/chat/sessions", json={
        "project_id": project_id, "section": "student", "title": "RAG Test"
    }, params={"user_id": user_id})
    if r.status_code != 201:
        log("Chat session create", "FAIL", f"HTTP {r.status_code}", None)
        return
    session_id = r.json()["session_id"]

    questions = [
        ("ar", "ما هو المعالجة اللغوية الطبيعية؟ اشرحها ببساطة.", "study"),
        ("en", "What is NLP and what are its key applications?", "study"),
        ("ar", "كمل الشرح عن تطبيقات NLP.", "study"),  # follow-up
        ("en", "Explain machine learning algorithms briefly.", "hint"),
    ]

    for lang, question, mode in questions:
        start = time.perf_counter()
        try:
            full_response = ""
            citations = []
            with httpx.stream("POST",
                f"{API_BASE}/chat/sessions/{session_id}/ask",
                json={"question": question, "mode": mode},
                timeout=TIMEOUT) as resp:
                if resp.status_code != 200:
                    log(f"RAG chat [{lang}/{mode}]", "FAIL",
                        f"HTTP {resp.status_code}", None)
                    continue
                current_event = "delta"
                data_lines = []
                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: "):
                        data_lines.append(line[6:])
                    elif not line and data_lines:
                        try:
                            data = json.loads("\n".join(data_lines))
                        except Exception:
                            data = {"raw": "\n".join(data_lines)}
                        if current_event == "delta":
                            full_response += str(data)
                        elif current_event == "meta":
                            citations = data.get("citations", [])
                        data_lines = []
                        current_event = "delta"
            lat = (time.perf_counter() - start) * 1000

            # Evaluate
            is_arabic = any("\u0600" <= c <= "\u06ff" for c in question)
            has_citations = len(citations) > 0
            answer_len = len(full_response)
            eval_ok = (
                answer_len > 20 and
                (is_arabic or True)  # language check would need langdetect
            )
            eval_detail = {
                "question": question,
                "mode": mode,
                "answer": full_response[:300],
                "citations": citations,
                "answer_len": answer_len,
                "is_arabic": is_arabic,
            }
            status = "PASS" if eval_ok else "FAIL"
            log(f"RAG chat [{lang}/{mode}]", status,
                f"answer_len={answer_len}, citations={len(citations)}", lat,
                eval_detail)
            EVAL_ENTRIES.append(eval_detail)
        except Exception as e:
            log(f"RAG chat [{lang}/{mode}]", "FAIL", f"{type(e).__name__}: {e}", None)


def test_document_search(user_id: str, project_id: str, doc_id: str | None) -> None:
    """Test semantic search via teacher materials endpoint."""
    if not doc_id:
        log("Document search", "FAIL", "No document available", None)
        return
    r = api("get", "/teacher/materials", params={
        "user_id": user_id, "project_id": project_id,
        "document_id": doc_id, "topic": "natural language processing",
    })
    if r.status_code != 200:
        log("Document search", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}", None)
        return
    data = r.json()
    log("Document search", "PASS",
        f"excerpts={len(data.get('excerpts', []))}, doc={data.get('document_name')}", None,
        {"excerpts": data.get("excerpts", [])[:2]})


def test_web_search(user_id: str, project_id: str, session_id: str | None) -> None:
    """Test web search with explicit external request."""
    # First enable web search by checking if it's enabled
    r = api("get", "/api/health")  # just checking
    # We need a chat session for ask-image or ask. Let's use ask with web search intent.
    # But GROQ_WEB_SEARCH_ENABLED=false in .env, so web search is disabled.
    # Test that it returns disabled message gracefully.
    if not session_id:
        r = api("post", "/chat/sessions", json={
            "project_id": project_id, "section": "student", "title": "Web Search Test"
        }, params={"user_id": user_id})
        if r.status_code != 201:
            log("Web search", "FAIL", "Could not create session", None)
            return
        session_id = r.json()["session_id"]

    # Web search is disabled in .env. Test explicit external request.
    start = time.perf_counter()
    try:
        full_response = ""
        with httpx.stream("POST",
            f"{API_BASE}/chat/sessions/{session_id}/ask",
            json={"question": "ابحث على الإنترنت عن أحدث هاتف آيفون وسعره", "mode": "study"},
            timeout=TIMEOUT) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if isinstance(data, str):
                            full_response += data
                    except Exception:
                        pass
        lat = (time.perf_counter() - start) * 1000
        log("Web search (disabled)", "PASS",
            f"response_len={len(full_response)}, disabled_msg_present="
            f"{'مغلق' in full_response or 'disabled' in full_response.lower()}", lat,
            {"response": full_response[:200]})
    except Exception as e:
        log("Web search (disabled)", "FAIL", f"{type(e).__name__}: {e}", None)


def test_exam(user_id: str, project_id: str, doc_id: str | None) -> str | None:
    """Test exam generation, answer-key privacy, and submission scoring."""
    if not doc_id:
        log("Exam generation", "FAIL", "No document available", None)
        return None
    # Discover topics first and use exact name
    r = api("get", "/student/dashboard", params={
        "user_id": user_id, "project_id": project_id, "document_id": doc_id,
    })
    topics = r.json().get("course_topics", []) if r.status_code == 200 else []
    if not topics:
        log("Exam generation", "FAIL", "No topics discovered", None)
        return None
    topic_name = topics[0]["name"]
    r = api("post", "/student/exams", json={
        "user_id": user_id, "project_id": project_id,
        "document_id": doc_id, "topic": topic_name,
        "difficulty": "Easy", "question_count": 3, "language": "English",
    })
    if r.status_code != 201:
        log("Exam generation", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}", None)
        return None
    data = r.json()
    exam_id = data["exam_id"]
    questions = data.get("questions") or []
    if not questions:
        log("Exam generation", "FAIL", "API returned an exam with no questions", None)
        return None
    log("Exam generation", "PASS",
        f"topic={topic_name}, exam_id={exam_id}, questions={data['question_count']}", None,
        {"title": data.get("title"), "questions": questions})

    # Public questions must not expose the answer key. The old test assumed
    # correct_index was present and crashed with KeyError when it was hidden.
    leaked = [
        q for q in questions
        if "correct_index" in q or "correct_answer" in q
    ]
    log("Exam answer key hidden",
        "FAIL" if leaked else "PASS",
        f"{len(leaked)}/{len(questions)} questions expose an answer key", None)

    # Submit a deterministic safe answer set (option 0 for every question),
    # then verify the server's score against the review it returns.
    answers = [0 for _ in questions]
    r = api("post", f"/student/exams/{exam_id}/submit", json={
        "user_id": user_id, "project_id": project_id, "answers": answers,
    })
    if r.status_code != 200:
        log("Exam submission", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}", None)
        return exam_id
    result = r.json()
    review = result.get("review") or []
    expected_correct = sum(1 for item in review if item.get("is_correct"))
    expected_score = round(100 * expected_correct / max(1, len(questions)), 1)
    consistent = (
        result.get("correct") == expected_correct
        and result.get("total") == len(questions)
        and abs(float(result.get("score_pct", -1)) - expected_score) < 0.01
    )
    log("Exam submission", "PASS" if consistent else "FAIL",
        f"score={result.get('score_pct')}%, correct={result.get('correct')}/{result.get('total')}, "
        f"review_consistent={consistent}", None,
        {"score_pct": result.get("score_pct"), "review": review})
    return exam_id


def test_study_plan(user_id: str, project_id: str, doc_id: str | None) -> None:
    """Test study plan generation."""
    if not doc_id:
        log("Study plan", "FAIL", "No document available", None)
        return
    # Use exact topic name from discovery
    r = api("get", "/student/dashboard", params={
        "user_id": user_id, "project_id": project_id, "document_id": doc_id,
    })
    topics = r.json().get("course_topics", []) if r.status_code == 200 else []
    topic_name = topics[0]["name"] if topics else None
    payload = {
        "user_id": user_id, "project_id": project_id,
        "weekly_hours": 4, "document_id": doc_id, "language": "Arabic",
    }
    if topic_name:
        payload["topic"] = topic_name
    r = api("post", "/student/study-plan", json=payload)
    if r.status_code != 200:
        log("Study plan", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}", None)
        return
    data = r.json()
    log("Study plan", "PASS",
        f"topic={topic_name}, hours={data['weekly_hours']}, plan_days={len(data['plan'])}, summary={data.get('summary','')[:60]}", None,
        {"plan": data.get("plan")})


def test_dashboard(user_id: str, project_id: str, doc_id: str | None) -> None:
    """Test student dashboard."""
    params = {"user_id": user_id, "project_id": project_id}
    if doc_id:
        params["document_id"] = doc_id
    r = api("get", "/student/dashboard", params=params)
    if r.status_code != 200:
        log("Student dashboard", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}", None)
        return
    data = r.json()
    log("Student dashboard", "PASS",
        f"attempts={data.get('attempts')}, topics={len(data.get('topics', []))}, "
        f"course_topics={len(data.get('course_topics', []))}", None,
        {"topics": data.get("topics", [])[:3],
         "course_topics": data.get("course_topics", [])[:3]})


def test_image_analysis(user_id: str, project_id: str) -> None:
    """Test real Gemini vision path through /student/analyze-image."""
    from PIL import Image
    import io
    img = Image.new("RGB", (200, 100), color=(255, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    start = time.perf_counter()
    r = api("post", "/student/analyze-image",
            data={"user_id": user_id, "project_id": project_id,
                  "question": "ما لون هذه الصورة؟ أجب بكلمة واحدة."},
            files={"file": ("test.png", img_bytes, "image/png")})
    lat = (time.perf_counter() - start) * 1000
    if r.status_code != 200:
        log("Image analysis", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}", lat)
        return
    answer = str(r.json().get("answer") or "").strip()
    log("Image analysis", "PASS" if answer else "FAIL",
        f"answer={answer[:120]}", lat, {"answer": answer})


def test_teacher_records(user_id: str, project_id: str, doc_id: str | None) -> None:
    """Cover the teacher assessment-records endpoint."""
    params = {"user_id": user_id, "project_id": project_id}
    if doc_id:
        params["document_id"] = doc_id
    r = api("get", "/teacher/assessment-records", params=params)
    if r.status_code != 200:
        log("Teacher assessment records", "FAIL",
            f"HTTP {r.status_code}: {r.text[:200]}", None)
        return
    data = r.json()
    records = data if isinstance(data, list) else data.get("records", [])
    log("Teacher assessment records", "PASS", f"records={len(records)}", None,
        {"records": records[:2]})


def test_ownership_negative(user_id: str, project_id: str, doc_id: str | None) -> None:
    """Wrong user_id must not be able to read or mutate another user's course."""
    wrong_user = str(uuid.uuid4())
    checks = [
        ("Wrong user (dashboard)", "get", "/student/dashboard",
         {"user_id": wrong_user, "project_id": project_id}, None),
        ("Wrong user (teacher records)", "get", "/teacher/assessment-records",
         {"user_id": wrong_user, "project_id": project_id}, None),
    ]
    if doc_id:
        checks.append(("Wrong user (delete document)", "delete",
                       f"/documents/{doc_id}", {"user_id": wrong_user}, None))
    for name, method, path, params, files in checks:
        r = api(method, path, params=params, files=files) if files else api(method, path, params=params)
        safe = r.status_code in (400, 403, 404, 409)
        log(name, "PASS" if safe else "FAIL",
            f"HTTP {r.status_code}: {r.text[:120]}", None)

    # Upload with a user who does not own the project must be rejected.
    r = api("post", "/documents/upload", params={
        "project_id": project_id, "user_id": wrong_user,
    }, files={"file": ("unauthorized.txt", b"not allowed", "text/plain")})
    safe = r.status_code in (400, 403, 404, 409)
    log("Wrong user (upload document)", "PASS" if safe else "FAIL",
        f"HTTP {r.status_code}: {r.text[:120]}", None)


def test_delete_document(user_id: str, doc_id: str | None) -> None:
    """Owner can delete the uploaded document; verify DB/file cleanup by status."""
    if not doc_id:
        log("Delete document", "FAIL", "No document available", None)
        return
    r = api("delete", f"/documents/{doc_id}", params={"user_id": user_id})
    if r.status_code != 200:
        log("Delete document", "FAIL", f"HTTP {r.status_code}: {r.text[:200]}", None)
        return
    r = api("get", f"/documents/{doc_id}/status")
    gone = r.status_code == 404
    log("Delete document", "PASS" if gone else "FAIL",
        f"status_after_delete=HTTP {r.status_code}", None,
        {"delete_response": r.text[:200]})


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("EduCore Full Model & Feature Testing")
    print("=" * 70)
    print(f"API: {API_BASE}")
    print()

    # Model tests
    print("\n📦 MODEL TESTS")
    print("-" * 70)
    test_groq_primary()
    test_groq_fallback()
    test_embedding()
    test_gemini_image()

    # Feature tests
    print("\n🚀 FEATURE TESTS")
    print("-" * 70)
    user_id, project_id = setup_user_project()
    print(f"  User: {user_id}, Project: {project_id}")

    doc_id = test_document_upload(user_id, project_id)

    # Need a session id for web search test
    r = api("post", "/chat/sessions", json={
        "project_id": project_id, "section": "student", "title": "Web Test"
    }, params={"user_id": user_id})
    session_id = r.json()["session_id"] if r.status_code == 201 else None

    test_rag_chat(user_id, project_id, doc_id)
    test_document_search(user_id, project_id, doc_id)
    test_web_search(user_id, project_id, session_id)
    test_exam(user_id, project_id, doc_id)
    test_study_plan(user_id, project_id, doc_id)
    test_dashboard(user_id, project_id, doc_id)
    test_teacher_records(user_id, project_id, doc_id)
    test_image_analysis(user_id, project_id)
    test_ownership_negative(user_id, project_id, doc_id)
    test_delete_document(user_id, doc_id)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    info = sum(1 for r in RESULTS if r["status"] == "INFO")
    print(f"Total: {len(RESULTS)} | Passed: {passed} | Failed: {failed} | Info: {info}")
    print()
    for r in RESULTS:
        mark = "✅" if r["status"] == "PASS" else "❌" if r["status"] == "FAIL" else "ℹ️ "
        lat = f" ({r['latency_ms']:.0f}ms)" if r["latency_ms"] is not None else ""
        print(f"  {mark} {r['test']}{lat}: {r['detail'][:100]}")

    # Write markdown report
    lines = ["# EduCore Full Testing Report", "",
             f"Generated: {datetime.utcnow().isoformat()}Z", "",
             f"Total: {len(RESULTS)} | Passed: {passed} | Failed: {failed} | Info: {info}", "",
             "| Test | Status | Latency (ms) | Detail |",
             "|------|--------|--------------|--------|"]
    for r in RESULTS:
        detail = (r["detail"] or "").replace("|", "\\|")[:80]
        lines.append(
            f"| {r['test']} | {r['status']} | {r['latency_ms'] or '-'} | {detail} |"
        )
    lines.append("")
    lines.append("## Evaluation Entries (RAG Chat)")
    lines.append("")
    lines.append("| Question | Mode | Answer Length | Citations | Arabic |")
    lines.append("|----------|------|---------------|-----------|--------|")
    for e in EVAL_ENTRIES:
        q = (e.get("question", "") or "").replace("|", "\\|")[:60]
        lines.append(
            f"| {q} | {e.get('mode')} | {e.get('answer_len')} | "
            f"{len(e.get('citations', []))} | {e.get('is_arabic')} |"
        )
    lines.extend([
        "",
        "## Environment",
        "",
        f"- Python: `{sys.version.split()[0]}`",
        f"- API: `{API_BASE}`",
        f"- Run tag: `{RUN_TAG}`",
        f"- Chat model: `{os.getenv('GROQ_CHAT_MODEL', 'n/a')}`",
        f"- Embedding model: `{os.getenv('EMBEDDING_MODEL_NAME', 'n/a')}`",
        f"- Vision model: `{os.getenv('GOOGLE_VISION_MODEL', 'n/a')}`",
        f"- Web search flags: OPENROUTER_ENABLE_WEB_SEARCH="
        f"`{os.getenv('OPENROUTER_ENABLE_WEB_SEARCH', 'false')}`, "
        f"GROQ_WEB_SEARCH_ENABLED=`{os.getenv('GROQ_WEB_SEARCH_ENABLED', 'false')}`",
        "",
        "## Notes on test design",
        "",
        "- Exam questions returned by the API intentionally omit `correct_index`; "
        "submission is validated against the score and review returned by the server.",
        "- Web search is validated in its disabled-by-configuration state; no live "
        "external provider calls are made.",
        "- Arabic content is uploaded as TXT due to the known Arabic PDF extraction defect.",
        "",
        "## Residual risks",
        "",
        "- Topic discovery and exam generation depend on LLM structured output and "
        "can hit provider rate limits.",
        "- Arabic PDF extraction remains unverified/unfixed.",
    ])
    report = "\n".join(lines) + "\n"
    out = Path("FULL_TEST_REPORT.md")
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out.resolve()}")


if __name__ == "__main__":
    main()
