
"""
EduCore — Streamlit Frontend (No Sign-in required).

Direct access:
- Toggle between Student & Teacher sections.
- Shared Knowledge Base (Upload, View and Delete documents).
- Conversational RAG with citations.
"""

from __future__ import annotations

import json
import os
import re

import httpx
import streamlit as st

from educore.rag.query_processor import process_query

# Configuration

API_BASE = os.getenv("EDUCORE_API_BASE", "http://127.0.0.1:8002/api")
TIMEOUT = httpx.Timeout(30.0, read=120.0)

DEFAULT_USER_EMAIL = "educore@example.com"


# Page config

st.set_page_config(
    page_title="EduCore — المنظومة التعليمية",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Session state

def init_state():
    defaults = {
        "user_id": None,
        "project_id": None,
        "chat_session_id": None,
        "messages": [],
        "section": "student",
        "student_ai_mode": "study",
        "delete_confirm": None,
        "practice_question": None,
        "practice_result": None,
        "student_exam": None,
        "student_exam_result": None,
        "student_exam_config": None,
        "student_chat_image_nonce": 0,
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()

if "bottom_section_switch" not in st.session_state:
    st.session_state.bottom_section_switch = st.session_state.section


def _on_bottom_section_change():
    selected_section = st.session_state.bottom_section_switch
    if selected_section != st.session_state.section:
        st.session_state.section = selected_section
        st.session_state.chat_session_id = None
        st.session_state.messages = []


def _enable_hint_mode():
    st.session_state.student_ai_mode = "hint"


st.markdown(
    """
    <style>
    .stAppViewBlockContainer { padding-bottom: 10rem; }
    .st-key-bottom_role_switcher {
        position: fixed; left: 0; right: 0; bottom: 0; z-index: 9999;
        padding: .55rem 1.2rem .45rem; background: rgba(20, 19, 18, .97);
        border-top: 1px solid rgba(255,255,255,.14); backdrop-filter: blur(10px);
    }
    .st-key-bottom_role_switcher [data-testid="stRadio"] > div {
        flex-direction: row; justify-content: center; gap: .6rem;
    }
    div[data-testid="stChatInput"] {
        position: fixed !important;
        left: max(20rem, 20vw) !important;
        right: 2rem !important;
        bottom: 5.25rem !important;
        z-index: 1000;
    }
    @media (max-width: 768px) {
        div[data-testid="stChatInput"] {
            left: 1rem !important;
            right: 1rem !important;
            bottom: 5rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# API helpers

def api_get(path: str, params: dict = None):
    try:
        response = httpx.get(
            f"{API_BASE}{path}",
            params=params,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        st.error(f"⚠️ خطأ في الاتصال بالسيرفر: {e}")
        return None


def api_post(
    path: str,
    json_data: dict = None,
    params: dict = None,
    files=None,
    data: dict = None,
):
    try:
        response = httpx.post(
            f"{API_BASE}{path}",
            json=json_data,
            params=params,
            files=files,
            data=data,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"⚠️ خطأ أثناء الطلب: {detail}")
        return None

    except Exception as e:
        st.error(f"⚠️ خطأ أثناء الطلب: {e}")
        return None


def api_delete(path: str):
    try:
        response = httpx.delete(
            f"{API_BASE}{path}",
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)

        st.error(f"⚠️ فشل حذف الملف: {detail}")
        return None

    except Exception as e:
        st.error(f"⚠️ خطأ أثناء حذف الملف: {e}")
        return None


def stream_sse(path: str, json_data: dict):
    """Stream SSE events from the API."""

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}{path}",
            json=json_data,
            timeout=TIMEOUT,
        ) as response:

            response.raise_for_status()
            current_event = "delta"
            data_lines = []

            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]

                elif line.startswith("data: "):
                    data_lines.append(line[6:])

                elif not line and data_lines:
                    raw_data = "\n".join(data_lines)
                    try:
                        data = json.loads(raw_data)
                    except (json.JSONDecodeError, TypeError):
                        data = raw_data
                    yield current_event, data
                    current_event = "delta"
                    data_lines = []

            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    data = json.loads(raw_data)
                except (json.JSONDecodeError, TypeError):
                    data = raw_data
                yield current_event, data

    except Exception as e:
        yield "error", str(e)


def render_citations(citations: list[dict]):
    if not citations:
        return

    with st.expander(f"📌 المصادر ({len(citations)})"):
        for citation in citations:
            source_number = citation.get("source_number", "?")
            title = citation.get("document_name", "مصدر")
            page = (
                f" · صفحة {citation['page_number']}"
                if citation.get("page_number")
                else ""
            )
            if citation.get("url"):
                st.markdown(
                    f"**[{source_number}] [{title}]({citation['url']})**{page}"
                )
            else:
                st.markdown(f"**[{source_number}] {title}{page}**")
            if citation.get("snippet"):
                st.caption(f"> _{citation['snippet']}_")


def normalize_chat_answer(
    content: str,
    citations: list[dict] | None,
    sentence_count: int | None = None,
) -> str:
    lower_content = (content or "").casefold()
    is_weekly_plan = (
        "دي خطة مذاكرتك" in lower_content
        or "here is your weekly study plan" in lower_content
    )
    if sentence_count is not None and not is_weekly_plan:
        parts = re.split(r"(?<=[.!?؟])\s+|\n+", (content or "").strip())
        content = " ".join(
            part.strip() for part in parts[:sentence_count] if part.strip()
        )

    citation_count = len(citations or [])
    content = re.sub(
        r"(?:\s*\[\d+\]\s*[.,;،؛:]?){3,}\s*$",
        "",
        content,
    ).rstrip()
    seen = set()

    def isolate_citation(match):
        number = int(match.group(1))
        if number < 1 or number > citation_count or number in seen:
            return ""
        seen.add(number)
        return f" \u2066[{number}]\u2069"

    content = re.sub(r"\s*\[(\d+)\]", isolate_citation, content)
    content = re.sub(r"[ \t]{2,}", " ", content)
    content = re.sub(r"\s+([.?!؟،؛])", r"\1", content)
    return content.strip()


# Automatic default setup

def ensure_default_user_and_project() -> bool:
    if st.session_state.user_id and st.session_state.project_id:
        return True

    user = api_post(
        "/users",
        {
            "name": "مستخدم النظام",
            "email": DEFAULT_USER_EMAIL,
            "role": "student",
        },
    )

    if user and "id" in user:
        st.session_state.user_id = user["id"]

        projects = api_get(
            "/projects",
            {"owner_id": user["id"]},
        )

        if projects and len(projects) > 0:
            st.session_state.project_id = projects[0]["id"]
            return True
        else:
            new_project = api_post(
                "/projects",
                {
                    "name": "المقرر الدراسي الأساسي",
                    "description": "المستودع المعرفي",
                },
                params={"owner_id": user["id"]},
            )

            if new_project and "id" in new_project:
                st.session_state.project_id = new_project["id"]
                return True
    return False

ensure_default_user_and_project()


# Sidebar

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=70,
    )

    st.title("🎓 EduCore")
    st.caption("منظومة RAG التعليمية المتكاملة")

    st.divider()
    section = st.session_state.section

    if section == "student":
        st.info(
            "💡 **قسم الطلاب**: شروحات مبنية على ملفات المقرر، "
            "وتدريب يساعدك تفكر خطوة بخطوة."
        )
        st.radio(
            "أسلوب مساعد الطالب",
            options=["study", "hint"],
            format_func=lambda value: (
                "📖 مذاكرة: شرح وتفسير" if value == "study"
                else "🧩 حل أو امتحان: تلميحات فقط"
            ),
            key="student_ai_mode",
        )
        if st.session_state.student_ai_mode == "hint":
            st.caption("أثناء الحل، هتاخد تلميح من المحاضرات من غير كشف الإجابة.")
    else:
        st.info(
            "💡 **قسم المدرسين**: إدارة المحاضرات ومساعدة في تنظيم "
            "المحتوى وإنشاء مواد تعليمية."
        )

    st.divider()
    if st.button(
        "🔄 بدء محادثة جديدة / New Chat",
        width="stretch",
    ):
        st.session_state.chat_session_id = None
        st.session_state.messages = []
        st.rerun()
    st.caption("المحادثات القديمة بتفضل محفوظة وتقدر تفتحها من قائمة المحادثات.")

    st.divider()
    saved_chats = []
    if st.session_state.user_id and st.session_state.project_id:
        saved_chats = api_get(
            "/chat/sessions",
            {
                "user_id": st.session_state.user_id,
                "project_id": st.session_state.project_id,
            },
        ) or []
    st.subheader(f"المحادثات السابقة ({len(saved_chats)})")
    if saved_chats:
        with st.container(height=360, border=False):
            for saved_chat in saved_chats:
                chat_id = saved_chat["id"]
                chat_title = (saved_chat.get("title") or "محادثة محفوظة").replace("\n", " ").strip()
                chat_section = saved_chat.get("section", "student")
                role_label = "طالب" if chat_section == "student" else "مدرّس"
                button_label = f"💬 {role_label} · {chat_title[:42]}{'…' if len(chat_title) > 42 else ''}"
                if st.button(
                    button_label,
                    key=f"open_saved_chat_{chat_id}",
                    type="primary" if chat_id == st.session_state.chat_session_id else "secondary",
                    width="stretch",
                    help=chat_title,
                ):
                    old_messages = api_get(f"/chat/sessions/{chat_id}/messages")
                    if old_messages is not None:
                        st.session_state.section = chat_section
                        st.session_state.bottom_section_switch = chat_section
                        st.session_state.chat_session_id = chat_id
                        st.session_state.messages = old_messages
                        st.rerun()
    else:
        st.caption("لسه مفيش محادثات محفوظة.")
# Main content

section_emoji = "👨‍🎓" if section == "student" else "👨‍🏫"

section_name = (
    "قسم الطلاب (Student Mode)"
    if section == "student"
    else "قسم المدرسين (Teacher Mode)"
)

st.subheader(f"{section_emoji} {section_name}")

tab_chat, tab_student, tab_practice, tab_docs = st.tabs(
    (
        [
            "💬 المحادثة التفاعلية / Chat",
            "🧑‍🏫 أدوات المدرّس / Teacher Workspace",
            "📖 دليل المدرّس / Guide",
            "📚 مستودع الملفات والمحاضرات / Knowledge Base",
        ]
        if section == "teacher"
        else [
            "💬 المحادثة التفاعلية / Chat",
            "📈 تقدم الطالب / Progress",
            "🎯 تدريب وامتحانات / Practice",
            "📚 مستودع الملفات والمحاضرات / Knowledge Base",
        ]
    )
)


# Documents tab

with tab_docs:

    st.markdown("##### 📤 رفع محاضرات أو كتب جديدة (PDF, TXT, CSV, XLSX)")

    uploaded_file = st.file_uploader(
        "اختر ملفاً لتدريبه داخل المنظومة",
        type=["pdf", "txt", "md", "csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded_file and st.button(
        "📤 رفع ومعالجة المستند",
        width="stretch",
    ):

        with st.spinner("جاري رفع واستخراج الـ Embeddings في الخلفية..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
            }

            result = api_post(
                "/documents/upload",
                params={"project_id": st.session_state.project_id},
                files=files,
            )

            if result:
                st.success(
                    f"✅ تم رفع **{result['filename']}** — "
                    "المعالجة جارية في الخلفية!"
                )
                st.rerun()

    st.divider()

    st.markdown("##### 📋 المحاضرات المتوفرة في المنظومة حالياً:")

    docs = api_get(
        "/documents",
        {"project_id": st.session_state.project_id},
    )

    if docs is not None and len(docs) > 0:

        for doc in docs:

            status_emoji = {
                "pending": "⏳ قيد الانتظار",
                "processing": "🔄 جاري الفهرسة",
                "indexed": "✅ جاهز ومفهرس",
                "failed": "❌ حدث خطأ",
            }.get(doc["status"], doc["status"])

            col1, col2, col3, col4 = st.columns(
                [3, 1.5, 1, 1]
            )

            with col1:
                st.write(f"📄 **{doc['filename']}**")

            with col2:
                st.write(status_emoji)

            with col3:
                st.write(f"📊 {doc['total_chunks']} chunks")

            with col4:

                confirm_key = f"confirm_delete_{doc['id']}"

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{doc['id']}",
                    width="stretch",
                ):
                    st.session_state.delete_confirm = doc["id"]
                    st.rerun()

            if doc["status"] == "failed" and doc.get("error_message"):
                st.error(f"تفاصيل فشل الفهرسة: {doc['error_message']}")

            # Confirmation area
            if st.session_state.delete_confirm == doc["id"]:

                st.warning(
                    f"هل أنت متأكد من حذف الملف: {doc['filename']}؟ "
                    "سيتم حذف محتواه والـ Embeddings المرتبطة به."
                )

                yes_col, no_col = st.columns(2)

                with yes_col:

                    if st.button(
                        "نعم، احذف الملف",
                        key=f"confirm_yes_{doc['id']}",
                        type="primary",
                        width="stretch",
                    ):

                        with st.spinner("جاري حذف الملف..."):

                            result = api_delete(
                                f"/documents/{doc['id']}"
                            )

                        if result:
                            st.session_state.delete_confirm = None
                            st.success(
                                f"تم حذف {result['filename']} بنجاح."
                            )
                            st.rerun()

                with no_col:

                    if st.button(
                        "إلغاء",
                        key=f"confirm_no_{doc['id']}",
                        width="stretch",
                    ):
                        st.session_state.delete_confirm = None
                        st.rerun()

    elif docs is not None:
        st.info(
            "لا توجد ملفات مرفوعة حالياً. "
            "ارفع أول محاضرة من الأعلى!"
        )


# Chat tab

with tab_chat:

    if section == "student":
        st.caption("اسأل عن مستواك في كل ملف، أو قل: اعملي خطة مذاكرة، وهسألك عن الملف والموضوع.")

    # Display the selected or current conversation.
    last_user_question = ""
    for msg in st.session_state.messages:

        role = msg.get("role", "user")
        content = msg.get("content", "")
        citations = msg.get("citations")
        if role == "user":
            last_user_question = content
        elif role == "assistant":
            requested_sentences = process_query(
                last_user_question
            ).requested_sentence_count if last_user_question else None
            content = normalize_chat_answer(
                content,
                citations,
                requested_sentences,
            )

        avatar = (
            "👤"
            if role == "user"
            else ("👨‍🎓" if section == "student" else "👨‍🏫")
        )

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            if role == "user" and msg.get("image_bytes"):
                st.image(msg["image_bytes"], caption="الصورة المرفقة في هذه الرسالة")
            if citations and role == "assistant":
                render_citations(citations)

    if not st.session_state.messages:
        st.caption("ابدأ محادثة جديدة أو اختار محادثة سابقة من القائمة. محادثاتك محفوظة حتى بعد بدء محادثة جديدة.")

    with st.expander("📷 أضف صورة تمرين إلى الشات", expanded=False):
        image_nonce = st.session_state.student_chat_image_nonce
        image_file = st.file_uploader(
            "ارفع صورة JPG أو PNG أو WebP",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"student_chat_image_file_{image_nonce}",
        )
        image_question = st.text_input(
            "تحب أساعدك فيها إزاي؟",
            value="اشرح التمرين في الصورة خطوة بخطوة.",
            key=f"student_chat_image_question_{image_nonce}",
        )
        if st.button("ابعت الصورة في الشات", key=f"send_chat_image_{image_nonce}"):
            if not image_file:
                st.warning("ارفع صورة الأول.")
            else:
                if not st.session_state.project_id or not st.session_state.user_id:
                    ensure_default_user_and_project()

                if not st.session_state.chat_session_id:
                    if not st.session_state.project_id or not st.session_state.user_id:
                        st.error("تعذر الاتصال بالسيرفر أو تحميل بيانات المقرر.")
                        st.stop()

                    chat = api_post(
                        "/chat/sessions",
                        {
                            "project_id": st.session_state.project_id,
                            "section": st.session_state.section,
                            "title": f"شرح صورة: {image_file.name}"[:72],
                        },
                        params={"user_id": st.session_state.user_id},
                    )
                    if chat:
                        st.session_state.chat_session_id = chat["session_id"]
                    else:
                        st.error("ماقدرتش أبدأ المحادثة. راجع اتصال السيرفر وحاول تاني.")

                if st.session_state.chat_session_id:
                    user_content = f"📷 {image_file.name}\n{image_question.strip() or 'اشرح التمرين في الصورة خطوة بخطوة.'}"
                    uploaded_image_bytes = image_file.getvalue()
                    with st.spinner("بفهم الصورة وبجهز الشرح..."):
                        image_result = api_post(
                            f"/chat/sessions/{st.session_state.chat_session_id}/ask-image",
                            files={
                                "file": (
                                    image_file.name,
                                    uploaded_image_bytes,
                                    image_file.type or "application/octet-stream",
                                )
                            },
                            data={
                                "question": image_question,
                                "mode": (
                                    st.session_state.student_ai_mode
                                    if section == "student" else "study"
                                ),
                            },
                        )
                    if image_result:
                        st.session_state.messages.append({
                            "role": "user",
                            "content": user_content,
                            "image_bytes": uploaded_image_bytes,
                        })
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": image_result["answer"],
                        })
                        st.session_state.student_chat_image_nonce += 1
                        st.rerun()

    # Create a database session only after the student or teacher sends the first message.
    if question := st.chat_input(
        "اسأل عن المحاضرات أو مستواك، أو قل: اعملي خطة مذاكرة..."
    ):
        if not st.session_state.project_id or not st.session_state.user_id:
            ensure_default_user_and_project()

        if not st.session_state.chat_session_id:
            if not st.session_state.project_id or not st.session_state.user_id:
                st.error("تعذر الاتصال بالسيرفر أو تحميل بيانات المقرر. يرجى إعادة المحاولة بعد ثوانٍ.")
                st.stop()

            chat = api_post(
                "/chat/sessions",
                {
                    "project_id": st.session_state.project_id,
                    "section": st.session_state.section,
                    "title": question.strip().replace("\n", " ")[:72] or "محادثة جديدة",
                },
                params={"user_id": st.session_state.user_id},
            )
            if chat:
                st.session_state.chat_session_id = chat["session_id"]
            else:
                st.error("ماقدرتش أبدأ المحادثة. راجع اتصال السيرفر وحاول تاني.")

        if st.session_state.chat_session_id:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user", avatar="👤"):
                st.markdown(question)

            with st.chat_message(
                "assistant",
                avatar=("👨‍🎓" if section == "student" else "👨‍🏫"),
            ):
                response_placeholder = st.empty()
                citations_placeholder = st.empty()
                full_response = ""
                citations_data = None
                stream_error = None

                for event_type, data in stream_sse(
                    f"/chat/sessions/{st.session_state.chat_session_id}/ask",
                    {
                        "question": question,
                        "mode": (
                            st.session_state.student_ai_mode
                            if section == "student"
                            else "study"
                        ),
                    },
                ):
                    if event_type == "meta":
                        try:
                            meta = data if isinstance(data, dict) else json.loads(data)
                            citations_data = meta.get("citations", [])
                        except Exception:
                            pass
                    elif event_type == "delta":
                        full_response += str(data)
                        response_placeholder.markdown(full_response + "▌")
                    elif event_type == "done":
                        response_placeholder.markdown(full_response)
                    elif event_type == "replace":
                        full_response = str(data)
                        response_placeholder.markdown(full_response)
                    elif event_type == "error":
                        stream_error = data
                        st.error(f"❌ خطأ: {data}")

                if citations_data:
                    with citations_placeholder.container():
                        render_citations(citations_data)

                if not full_response.strip():
                    full_response = (
                        "الرد ماوصلش بسبب مشكلة اتصال بالنموذج. سؤالك محفوظ في المحادثة؛ "
                        "ابعت نفس السؤال تاني بعد لحظة."
                    )
                    response_placeholder.warning(full_response)
                elif stream_error:
                    full_response += "\n\nالرد اتقطع قبل ما يكتمل. تقدر تبعت السؤال تاني."
                    response_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "citations": citations_data,
                    }
                )
            # Render the complete new turn above the input on the next pass.
            st.rerun()
# Student AI — per-file progress dashboard

with tab_student:

    st.subheader("🧑‍🏫 أدوات المدرّس" if section != "student" else "📈 تقدم الطالب")
    if section == "student":
        st.caption("مستواك بيتحسب من محاولاتك المسجلة لكل ملف. ولخطة مذاكرة، اطلبها من الشات الرئيسي.")

    if section != "student":
        from educore.teacher_ai.ui import render_teacher_workspace

        render_teacher_workspace(
            user_id=st.session_state.user_id,
            project_id=st.session_state.project_id,
            api_get=api_get,
        )

    elif st.session_state.user_id and st.session_state.project_id:
        progress_documents = api_get(
            "/documents", {"project_id": st.session_state.project_id}
        ) or []
        progress_files = [
            {"id": item["id"], "filename": item["filename"]}
            for item in progress_documents
            if item.get("status") == "indexed" and int(item.get("total_chunks") or 0) > 0
        ]
        progress_file_ids = [item["id"] for item in progress_files]
        if st.session_state.get("student_progress_document_id") not in ["__all_files__", *progress_file_ids]:
            st.session_state.student_progress_document_id = (
                progress_file_ids[0] if progress_file_ids else "__all_files__"
            )
        progress_document_id = st.selectbox(
            "اعرض مستواك في",
            options=["__all_files__", *progress_file_ids],
            format_func=lambda value: (
                "كل ملفات المادة" if value == "__all_files__"
                else next((item["filename"] for item in progress_files if item["id"] == value), value)
            ),
            key="student_progress_document_id",
            disabled=not progress_file_ids,
        )
        progress_params = {
            "user_id": st.session_state.user_id,
            "project_id": st.session_state.project_id,
        }
        if progress_document_id != "__all_files__":
            progress_params["document_id"] = progress_document_id
        progress = api_get("/student/dashboard", progress_params)

        if progress:
            if progress_document_id != "__all_files__":
                selected_progress_file = next(
                    (item["filename"] for item in progress_files if item["id"] == progress_document_id),
                    "الملف المختار",
                )
                st.caption(f"التقدم المعروض مبني على محاولاتك في: **{selected_progress_file}**")
            elif progress_files:
                st.caption("اختار ملفًا من القائمة لعرض مستواك ونتائجك فيه لوحده.")
            metric_attempts, metric_accuracy = st.columns(2)
            metric_attempts.metric("المحاولات المسجلة", progress["attempts"])
            accuracy = progress.get("overall_accuracy_pct")
            metric_accuracy.metric(
                "دقة الإجابات",
                f"{accuracy}%" if accuracy is not None else "لسه مفيش بيانات",
            )

            if not progress.get("course_topics"):
                st.info("ارفع ملفات المدرّس وفهرسها عشان تظهر موضوعات المادة هنا.")
            st.markdown("#### مستوى موضوعات الملف المختار")
            for item in progress.get("topics", []):
                with st.container(border=True):
                    title_col, score_col = st.columns([3, 1])
                    title_col.markdown(f"**{item['topic']}** · {item['skill']}")
                    status_labels = {
                        "Insufficient data": "بيانات غير كافية",
                        "Needs support": "محتاج تركيز",
                        "Developing": "في تقدم",
                        "On track": "جيد",
                    }
                    attempts_count = item["attempts_count"]
                    title_col.caption(
                        f"{attempts_count} محاولات · {status_labels.get(item['status'], item['status'])}"
                    )
                    score_col.metric(
                        "الإتقان",
                        f"{item['bkt_mastery_pct']}%" if attempts_count else "—",
                    )
                    if attempts_count:
                        st.progress(min(1.0, max(0.0, item["bkt_mastery_pct"] / 100)))
                    else:
                        st.caption("ابدأ بالتدريب على الموضوع عشان يظهر تقدير مستواك.")
                    if item["observed_accuracy_pct"] is not None:
                        st.caption(f"دقة الإجابات في الموضوع: {item['observed_accuracy_pct']}%")

            weak_topics = progress.get("weakest_topics", [])
            if weak_topics:
                st.warning(
                    "ابدأ بالموضوع الأضعف حاليًا: "
                    f"**{weak_topics[0]['topic']}** "
                    f"({weak_topics[0]['bkt_mastery_pct']}% إتقان)."
                )
            elif progress["attempts"] == 0:
                st.info("حل بعض الأسئلة من تبويب التدريب عشان نبدأ تكوين صورة عن مستواك.")

        else:
            st.info("ابدأ بحل سؤال من تبويب التدريب، وبعدها هتظهر بيانات تقدمك هنا.")
    else:
        st.info("جاري إعداد حساب الطالب والمقرر.")


# Student AI — adaptive practice and RAG-grounded exams

with tab_practice:

    st.subheader("🎯 تدريب وامتحانات")
    st.caption("التدريب والامتحان بيتولدوا من ملفات المقرر المفهرسة. ارفع المحاضرة الأول لو مفيش مصدر مناسب.")

    if section != "student":
        st.info("أدوات تحضير الدروس والامتحانات والتصحيح والتحليلات موجودة في تبويب «أدوات المدرّس».")

    elif st.session_state.user_id and st.session_state.project_id:
        uploaded_documents = api_get(
            "/documents", {"project_id": st.session_state.project_id}
        ) or []
        course_files = [
            {
                "id": item["id"],
                "filename": item["filename"],
                "total_chunks": item.get("total_chunks", 0),
            }
            for item in uploaded_documents
            if item.get("status") == "indexed" and int(item.get("total_chunks") or 0) > 0
        ]
        file_ids = [item["id"] for item in course_files]
        if st.session_state.get("practice_document_id") not in file_ids:
            st.session_state.practice_document_id = file_ids[0] if file_ids else None
        st.markdown("#### سؤال تدريب تكيفي")
        if file_ids:
            practice_document_id = st.selectbox(
                "اختار الملف اللي هتتدرب منه",
                options=file_ids,
                format_func=lambda value: next(
                    (item["filename"] for item in course_files if item["id"] == value), value
                ),
                key="practice_document_id",
            )
        else:
            practice_document_id = None
            st.info("مفيش ملفات مفهرسة للتدريب. ارفع ملفًا من مستودع الملفات الأول.")
        selected_course_data = (
            api_get(
                "/student/dashboard",
                {
                    "user_id": st.session_state.user_id,
                    "project_id": st.session_state.project_id,
                    "document_id": practice_document_id,
                },
            )
            if practice_document_id else None
        )
        selected_course_topics = selected_course_data.get("course_topics", []) if selected_course_data else []
        topic_names = [item["name"] for item in selected_course_topics]
        if st.session_state.get("practice_topic_choice") not in [None, *topic_names]:
            st.session_state.pop("practice_topic_choice", None)
        current_practice = st.session_state.get("practice_question")
        if current_practice and (
            current_practice.get("topic") not in topic_names
            or current_practice.get("document_id") != practice_document_id
        ):
            st.session_state.practice_question = None
            st.session_state.practice_result = None
        current_exam = st.session_state.get("student_exam")
        if current_exam and current_exam.get("document_id") not in file_ids:
            st.session_state.student_exam = None
            st.session_state.student_exam_result = None
        if file_ids and not topic_names:
            st.info("موضوعات التدريب هتظهر من ملفات المدرّس بعد رفعها وفهرستها في مستودع الملفات.")

        topic_choice = st.selectbox(
            "اختار الموضوع من الملف أو سيبه للنظام حسب مستواك",
            options=[None, *topic_names],
            format_func=lambda value: "اختيار تلقائي حسب مستواي داخل الملف" if value is None else value,
            key="practice_topic_choice",
            disabled=not topic_names,
        )

        if st.button(
            "اختارلي سؤال",
            key="get_practice_question",
            disabled=not topic_names or not practice_document_id,
        ):
            practice_question = api_post(
                "/student/practice-question",
                {
                    "user_id": st.session_state.user_id,
                    "project_id": st.session_state.project_id,
                    "document_id": practice_document_id,
                    "topic": topic_choice,
                },
            )
            if practice_question:
                old_question = st.session_state.practice_question
                if old_question:
                    st.session_state.pop(
                        f"practice_answer_{old_question['question_id']}", None
                    )
                st.session_state.practice_question = practice_question
                st.session_state.practice_result = None

        practice_question = st.session_state.practice_question
        if practice_question:
            st.markdown(f"**{practice_question['topic']} · {practice_question['difficulty']}**")
            with st.form(f"practice_form_{practice_question['question_id']}"):
                selected_index = st.radio(
                    practice_question["question"],
                    options=[None, *range(len(practice_question["options"]))],
                    format_func=lambda index: "اختار إجابة" if index is None else practice_question["options"][index],
                    index=0,
                    key=f"practice_answer_{practice_question['question_id']}",
                )
                submitted = st.form_submit_button("تحقق من إجابتي", type="primary")

            if submitted:
                if selected_index is None:
                    st.warning("اختار إجابة الأول.")
                else:
                    result = api_post(
                        f"/student/exams/{practice_question['exam_id']}/submit",
                        {
                            "user_id": st.session_state.user_id,
                            "project_id": st.session_state.project_id,
                            "answers": [selected_index],
                        },
                    )
                    if result:
                        review = result["review"][0]
                        st.session_state.practice_result = {
                            "is_correct": review["is_correct"],
                            "correct_answer": review["correct_answer"],
                            "explanation": review["explanation"],
                        }
                        st.session_state.practice_question = None

        practice_result = st.session_state.practice_result
        if practice_result:
            if practice_result["is_correct"]:
                st.success("إجابة صحيحة! أحسنت.")
            else:
                st.error(f"الإجابة الصحيحة: {practice_result['correct_answer']}")
            st.info(practice_result["explanation"])
            st.caption("اتسجلت المحاولة واتحدث تقدير الإتقان في لوحة التقدم.")

        st.divider()
        st.markdown("#### إنشاء امتحان")
        with st.popover("🧩 إزاي أستخدم التلميحات؟"):
            st.write(
                "فعّل وضع **حل أو امتحان: تلميحات فقط** من قائمة الطالب على الشمال، "
                "وبعدها اسأل في تبويب المحادثة عن السؤال أو الجزء اللي وقفت فيه. "
                "المساعد هيدي تلميحًا من الملف من غير ما يكشف الإجابة النهائية."
            )
            st.button(
                "فعّل وضع التلميح",
                key="enable_exam_hint_mode",
                on_click=_enable_hint_mode,
                width="stretch",
            )
        filename_counts = {}
        for item in course_files:
            filename_counts[item["filename"]] = filename_counts.get(item["filename"], 0) + 1
        filename_seen = {}
        file_labels = {}
        for item in course_files:
            filename = item["filename"]
            filename_seen[filename] = filename_seen.get(filename, 0) + 1
            duplicate_label = (
                f" — نسخة {filename_seen[filename]}"
                if filename_counts[filename] > 1
                else ""
            )
            file_labels[item["id"]] = f"{filename}{duplicate_label}"
        if st.session_state.get("exam_document_id") not in file_ids:
            st.session_state.exam_document_id = file_ids[0] if file_ids else None
        exam_document_id = st.selectbox(
            "اختار ملف المحاضرة",
            options=file_ids if file_ids else [None],
            format_func=lambda value: next(
                (label for file_id, label in file_labels.items() if file_id == value),
                "ارفع وفهرس ملفًا من مستودع المدرّس أولًا",
            ),
            key="exam_document_id",
            disabled=not file_ids,
        )
        selected_file_data = (
            api_get(
                "/student/dashboard",
                {
                    "user_id": st.session_state.user_id,
                    "project_id": st.session_state.project_id,
                    "document_id": exam_document_id,
                },
            )
            if exam_document_id else None
        )
        exam_topics = (
            selected_file_data.get("course_topics", []) if selected_file_data else []
        )
        exam_topic_names = [item["name"] for item in exam_topics]
        if st.session_state.get("exam_topic") not in exam_topic_names:
            st.session_state.pop("exam_topic", None)
        exam_topic = st.selectbox(
            "اختار الموضوع جوه الملف",
            options=exam_topic_names if exam_topic_names else [None],
            format_func=lambda value: value or "جاري استخراج الموضوعات من الملف المختار",
            key="exam_topic",
            disabled=not exam_topic_names,
        )
        if file_ids and not exam_topic_names:
            st.warning("مش قادر أحدد موضوعات الملف حاليًا. تأكد إن الملف مفهرس وفيه نص واضح، وبعدها جرّب تاني.")
        exam_language = st.selectbox(
            "لغة الامتحان والتفسير",
            options=["Arabic", "English"],
            format_func=lambda value: "العربية" if value == "Arabic" else "English",
            key="exam_language",
        )
        exam_difficulty = st.selectbox(
            "مستوى الصعوبة",
            ["Mixed", "Easy", "Medium", "Hard"],
            format_func=lambda value: "متنوع" if value == "Mixed" else value,
            key="exam_difficulty",
        )
        exam_question_type = st.selectbox("Question type", ["mcq", "true_false", "short_answer", "essay", "mixed"],
                                          format_func=lambda value: {"mcq": "Multiple choice", "true_false": "True / False", "short_answer": "Short answer", "essay": "Essay", "mixed": "Mixed"}[value],
                                          key="exam_question_type")
        question_count = st.slider(
            "عدد الأسئلة", min_value=1, max_value=20, value=5, key="exam_question_count"
        )
        exam_config = (
            exam_document_id,
            exam_topic,
            exam_language,
            exam_difficulty,
            question_count,
            exam_question_type,
        )
        if st.session_state.get("student_exam_config") != exam_config:
            st.session_state.student_exam = None
            st.session_state.student_exam_result = None
        if st.button(
            "أنشئ الامتحان",
            key="generate_student_exam",
            disabled=not exam_topic_names,
        ):
            previous_exam = st.session_state.student_exam
            if previous_exam:
                for index in range(len(previous_exam.get("questions", []))):
                    st.session_state.pop(
                        f"exam_answer_{previous_exam['exam_id']}_{index}", None
                    )
            st.session_state.student_exam = None
            st.session_state.student_exam_result = None
            st.session_state.student_exam_config = None
            exam_request = {
                    "user_id": st.session_state.user_id,
                    "project_id": st.session_state.project_id,
                    "document_id": exam_document_id,
                    "topic": exam_topic,
                    "difficulty": exam_difficulty,
                    "question_count": question_count,
                    "language": exam_language,
                    "question_type": exam_question_type,
            }
            exam = api_post("/student/exams", exam_request)
            if exam:
                st.session_state.student_exam = exam
                st.session_state.student_exam_result = None
                st.session_state.student_exam_config = exam_config

        exam = st.session_state.student_exam
        if exam:
            st.markdown(f"### {exam['title']}")
            st.caption("المصدر: ملفات المقرر المفهرسة من المدرّس")
            if exam.get("citations"):
                with st.expander("المصادر المستخدمة"):
                    for citation in exam["citations"]:
                        page = f" · صفحة {citation['page_number']}" if citation.get("page_number") else ""
                        st.caption(f"{citation['document_name']}{page}")

            for index, question in enumerate(exam["questions"]):
                with st.container(border=True):
                    question_col, hint_col = st.columns([5, 1.5])
                    question_col.markdown(f"**{index + 1}. {question['question']}**")
                    hints = list(question.get("hints") or [])
                    answer_revealed = bool(question.get("answer_revealed"))
                    if answer_revealed:
                        hint_col.button(
                            "اكتملت التلميحات",
                            key=f"exam_hint_done_{exam['exam_id']}_{question['question_id']}",
                            disabled=True,
                            width="stretch",
                        )
                    elif hint_col.button(
                        f"💡 تلميح AI ({len(hints) + 1}/3)",
                        key=f"exam_hint_{exam['exam_id']}_{question['question_id']}",
                        width="stretch",
                    ):
                        with st.spinner("الـAI بيجهز التلميح من المحاضرة..."):
                            hint_result = api_post(
                                f"/student/exams/{exam['exam_id']}/questions/{index}/hint",
                                {
                                    "user_id": st.session_state.user_id,
                                    "project_id": st.session_state.project_id,
                                },
                            )
                        if hint_result:
                            question["hints"] = hint_result.get("hints", hints)
                            question["hints_used"] = hint_result.get(
                                "hints_used", len(question["hints"])
                            )
                            question["answer_revealed"] = hint_result.get(
                                "answer_revealed", False
                            )
                            if hint_result.get("answer_revealed"):
                                question["revealed_answer"] = hint_result.get(
                                    "correct_answer", ""
                                )
                                question["revealed_explanation"] = hint_result.get(
                                    "explanation", ""
                                )
                            st.session_state.student_exam = exam

                    answer_key = f"exam_answer_{exam['exam_id']}_{index}"
                    kind = question.get("question_type", "mcq")
                    if kind == "mcq":
                        st.radio("Choose your answer", options=[None, *range(len(question["options"]))],
                                 format_func=lambda choice, options=question["options"]: "Choose" if choice is None else options[choice], index=0, key=answer_key)
                    elif kind == "true_false":
                        st.radio("Your answer", options=[None, True, False], format_func=lambda value: "Choose" if value is None else ("True" if value else "False"), index=0, key=answer_key)
                    elif kind == "essay":
                        st.text_area("Your answer", key=answer_key, height=140)
                    else:
                        st.text_input("Your answer", key=answer_key)
                    for hint_number, hint_text in enumerate(
                        question.get("hints") or [], start=1
                    ):
                        st.info(f"💡 التلميح {hint_number}/3: {hint_text}")
                    if question.get("answer_revealed"):
                        st.success(
                            f"الإجابة: {question.get('revealed_answer', '')}"
                        )
                        if question.get("revealed_explanation"):
                            st.write(question["revealed_explanation"])

            submit_exam = st.button(
                "سلّم الامتحان",
                key=f"submit_exam_{exam['exam_id']}",
                type="primary",
            )

            if submit_exam:
                answers = [
                    st.session_state.get(f"exam_answer_{exam['exam_id']}_{index}")
                    for index in range(len(exam["questions"]))
                ]
                result = api_post(
                    f"/student/exams/{exam['exam_id']}/submit",
                    {
                        "user_id": st.session_state.user_id,
                        "project_id": st.session_state.project_id,
                        "answers": answers,
                    },
                )
                if result:
                    st.session_state.student_exam_result = result
                    st.session_state.student_exam = None

        exam_result = st.session_state.student_exam_result
        if exam_result:
            if exam_result.get("score_pct") is None:
                st.info(f"Exam submitted. {exam_result.get('ungraded_count', 0)} essay response(s) need review.")
            else:
                st.success(f"Result: {exam_result['score_pct']}% ({exam_result['correct']} correct of {exam_result.get('graded_count', exam_result['total'])} graded)")
            with st.expander("مراجعة الإجابات", expanded=True):
                for index, item in enumerate(exam_result["review"], start=1):
                    status = "??" if item["is_correct"] is None else "?" if item["is_correct"] else "?"
                    st.markdown(f"**{status} {index}. {item['question']}**")
                    st.caption(f"إجابتك: {item['your_answer']} · الصحيح: {item['correct_answer']}")
                    st.write(item["explanation"])

    else:
        st.info("جاري إعداد حساب الطالب والمقرر.")

with st.container(key="bottom_role_switcher"):
    st.radio(
        "التبديل بين الأقسام",
        options=["student", "teacher"],
        format_func=lambda value: (
            "👨‍🎓 الطالب" if value == "student" else "👨‍🏫 المدرّس"
        ),
        horizontal=True,
        key="bottom_section_switch",
        on_change=_on_bottom_section_change,
    )
