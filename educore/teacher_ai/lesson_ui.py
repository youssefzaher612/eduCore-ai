"""
EduCore Teacher AI - Lesson preparation interface.

Run:  streamlit run app.py

The UI only talks to the module's public interface:
    LessonRequest -> LessonGenerator.generate() -> LessonResult
and renders the lesson, the deterministic verification results and the quality review.
API keys and stack traces are never shown; errors go through presentation.friendly_error.
"""

import json
import logging

import streamlit as st

from educore.teacher_ai.core import config
from educore.teacher_ai.core.llm_client import LLMError
from educore.teacher_ai.modules.lesson_preparation.generator import LessonGenerator
from educore.teacher_ai.modules.lesson_preparation.service import LessonPreparationService
from educore.teacher_ai.modules.lesson_preparation.presentation import (STATUS_LABELS, friendly_error,
                                                     group_issues_by_section, lesson_to_markdown,
                                                     section_title, section_to_markdown)
from educore.teacher_ai.modules.lesson_preparation.quality_reviewer import QualityReviewer
from educore.teacher_ai.modules.lesson_preparation.schemas import REQUIRED_SECTIONS, LessonRequest

logger = logging.getLogger("educore.teacher.lesson_ui")

SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
SOURCE_LABEL = {
    "deterministic": "automatic check",
    "structure": "structure check",
    "ai_review": "AI review",
    "model_marker": "marked by the generator",
}
MODE_HELP = {
    "Prepare my lesson": "Teaching methodology and delivery: scripts, board work, examples, checks.",
    "Understand and prepare": "First deepens your own understanding of the topic, then gives the full teaching guide.",
}
SUBJECT_EXAMPLES = "e.g. Mathematics, Chemistry, Physics, History, English, Computer Science"


def _esc(text) -> str:
    return str(text).replace("$", "\\$")


def render_issue(issue) -> None:
    where = issue.to_dict()["path_text"] or "whole lesson"
    source = SOURCE_LABEL.get(issue.source, issue.source)
    body = f"**{_esc(where)}** · _{source}_\n\n{_esc(issue.message)}"
    if issue.suggestion:
        body += f"\n\n**Suggestion:** {_esc(issue.suggestion)}"
    {"critical": st.error, "warning": st.warning}.get(issue.severity, st.info)(body, icon=SEVERITY_ICON.get(issue.severity))





def render_issue_list(issues, empty_text) -> None:
    if not issues:
        st.caption(empty_text)
        return
    for severity in ("critical", "warning", "info"):
        for issue in (i for i in issues if i.severity == severity):
            render_issue(issue)


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

def request_form(rag_topic: str = "", source_file: str = ""):
    """Return a LessonRequest when the form is submitted and valid, else None."""
    with st.form("lesson_form"):
        st.subheader("Lesson details")
        c1, c2 = st.columns(2)
        subject = c1.text_input("Subject", placeholder=SUBJECT_EXAMPLES, max_chars=100)
        grade = c2.text_input("Grade / level", placeholder="e.g. Grade 9", max_chars=50)
        topic = st.text_input(
            "Topic from the selected course file",
            value=rag_topic,
            placeholder="Choose a topic from the selected file above",
            max_chars=200,
            disabled=bool(rag_topic),
        )
        if source_file:
            st.caption(f"Course source: {source_file}")

        mode = st.radio("Preparation mode", list(MODE_HELP), horizontal=True,
                        captions=list(MODE_HELP.values()))
        c3, c4, c5 = st.columns(3)
        language = c3.selectbox("Lesson language", ["English", "Arabic", "Egyptian Arabic"])
        duration = c4.number_input("Duration (minutes)", min_value=15, max_value=180, value=45, step=5)
        questions = c5.number_input("Practice questions", min_value=1, max_value=20, value=3, step=1)
        c6, c7 = st.columns(2)
        level = c6.selectbox("Student level", ["Mixed", "Beginner", "Intermediate", "Advanced"])
        style = c7.selectbox("Teaching style", ["Interactive", "Visual", "Practical", "Traditional", "Mixed"])
        extra = st.text_area("Extra instructions (optional)", max_chars=1000,
                             placeholder="Focus areas, classroom context, constraints…")
        use_ai = st.checkbox("Include AI quality review", value=config.ENABLE_LLM_REVIEW,
                             help="A second AI pass looks for contradictions and weak sections. "
                                  "Calculations are always checked by computation, not by AI.")
        submitted = st.form_submit_button("Prepare lesson", type="primary")

    if not submitted:
        return None
    st.session_state["use_ai"] = use_ai
    try:
        return LessonRequest(
            subject=subject, grade=grade, topic=topic, preparation_mode=mode, language=language,
            duration_minutes=int(duration), student_level=level, teaching_style=style,
            question_count=int(questions), extra_instructions=extra)
    except Exception as exc:  # pydantic ValidationError
        problems = []
        errors = getattr(exc, "errors", None)
        if callable(errors):
            for err in errors():
                field = ".".join(str(p) for p in err.get("loc", ()))
                problems.append(f"**{field.replace('_', ' ').capitalize()}**: please enter a valid value.")
        st.error("Please complete the form:\n\n" + "\n".join(f"- {p}" for p in problems or ["Some values are invalid."]))
        return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(request, source_material: str = "") -> None:
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    service = LessonPreparationService(
        generator=LessonGenerator(), reviewer=QualityReviewer(use_llm=st.session_state.get("use_ai", True)))
    try:
        with st.status("Preparing your lesson…", expanded=True) as status:
            st.write("• Writing the lesson, then checking calculations and quality…")
            prepared = service.prepare(request, source_material=source_material)
            status.update(label="Lesson ready", state="complete", expanded=False)
        st.session_state["result"] = prepared
    except LLMError as exc:
        logger.warning("lesson preparation stopped by the AI provider: %s", exc.message)
        st.session_state["error"] = friendly_error(exc)
    except Exception as exc:
        logger.warning("AI generation task=teacher_lesson stage=failed error_category=%s", type(exc).__name__)
        st.session_state["error"] = friendly_error(exc)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_result(prepared) -> None:
    report = prepared.report
    status = report.status
    banner = {"ready": st.success, "needs_attention": st.warning, "blocked": st.error}.get(status, st.info)
    banner(f"**{STATUS_LABELS.get(status, status)}**")
    if status == "blocked":
        st.caption("Automatic checks found errors in the lesson. Review the red items before teaching anything from it.")
    if prepared.was_repaired:
        st.info("Some items failed verification and were corrected automatically, then verified again.", icon="🛠️")

    m1, m2, m3 = st.columns(3)
    m1.metric("Critical errors", len(report.critical))
    m2.metric("Warnings", len(report.warnings))
    m3.metric("Notes", len(report.info))

    tab_lesson, tab_verify, tab_review, tab_export = st.tabs(
        ["Lesson", "Calculation checks", "Quality review", "Download"])
    by_section = group_issues_by_section(report.issues)
    lesson = prepared.lesson

    with tab_lesson:
        st.header(_esc(lesson.get("lesson_title", "Lesson plan")))
        for key in REQUIRED_SECTIONS:
            if key == "lesson_title" or key not in lesson:
                continue
            value = lesson[key]
            if key == "teacher_learning":
                if not (isinstance(value, dict) and value.get("include")):
                    continue
                value = {k: v for k, v in value.items() if k != "include"}
            section_issues = by_section.get(key, [])
            badge = ""
            if section_issues:
                worst = "critical" if any(i.severity == "critical" for i in section_issues) else \
                        "warning" if any(i.severity == "warning" for i in section_issues) else "info"
                badge = f"  {SEVERITY_ICON[worst]} {len(section_issues)}"
            with st.expander(section_title(key) + badge, expanded=key in ("lesson_overview", "lesson_flow")):
                for issue in section_issues:
                    render_issue(issue)
                st.markdown(section_to_markdown(key, value))

    with tab_verify:
        st.caption("Mathematics and chemistry equations are checked by computation, not by AI. "
                   "These results cannot be overridden by the AI review.")
        outcome = report.verification
        if outcome is not None:
            st.write(f"**Status:** {outcome.status.replace('_', ' ')}")
            if outcome.ran:
                st.write("**Checkers run:** " + ", ".join(outcome.ran))
            for name, reason in outcome.skipped.items():
                st.write(f"**Skipped:** {name} ({reason})")
            render_issue_list(outcome.issues, "No calculation or equation problems were found.")
            counts_by_status = outcome.report.status_counts()
            if outcome.report.records:
                st.write("**Items checked:** " + ", ".join(
                    f"{n} {s.replace('_', ' ')}" for s, n in counts_by_status.items() if n))
                with st.expander("Coverage per item"):
                    for record in outcome.report.records:
                        line = f"**{_esc(record.label)}** — {record.status.replace('_', ' ')}"
                        if record.passed:
                            line += "\n\n" + "; ".join(_esc(p) for p in record.passed)
                        if record.notes:
                            line += "\n\n_Not checked:_ " + "; ".join(_esc(n) for n in record.notes)
                        st.markdown(line)
            for note in outcome.report.limitations:
                st.caption("Limit: " + note)

    with tab_review:
        st.write(f"**AI review:** {_esc(report.ai_message)}")
        if report.omitted_for_review:
            st.caption("Left out of the AI review to fit its size limit: " + ", ".join(report.omitted_for_review))
        render_issue_list(report.issues, "No quality problems were found.")
        for strength in report.strengths:
            st.success(_esc(strength), icon="✅")
        repair = prepared.repair
        if repair and (repair.accepted or repair.rejected):
            with st.expander("Automatic corrections"):
                for item in repair.accepted:
                    st.write(f"Corrected and re-verified: {item}")
                for item, reason in repair.rejected:
                    st.write(f"Left unchanged: {item} ({_esc(reason)})")

    with tab_export:
        st.download_button("Download lesson (Markdown)", lesson_to_markdown(lesson), file_name="lesson_plan.md",
                           mime="text/markdown")
        st.download_button("Download full report (JSON)",
                           json.dumps(prepared.to_dict(), ensure_ascii=False, indent=2),
                           file_name="lesson_report.json", mime="application/json")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render_lesson_page(
    source_material: str = "",
    selected_topic: str = "",
    source_file: str = "",
) -> None:
    st.title("🎓 EduCore · تحضير الدروس")
    st.caption("Lesson preparation grounded in the selected indexed course file, with calculations checked by computation.")

    if not source_material.strip() or not selected_topic:
        st.info("Choose an indexed course file and one of its topics above to prepare a RAG-grounded lesson.")
        return
    st.info(f"Lesson topic: **{selected_topic}** · RAG source: **{source_file}**")

    if config.settings.active_provider != "ollama" and not config.OPENROUTER_API_KEY:
        st.warning(f"No {config.settings.active_provider.upper()} API key is configured. Add it to EduCore's .env file and restart the app.")

    request = request_form(rag_topic=selected_topic, source_file=source_file)
    if request is not None:
        run_generation(request, source_material=source_material)

    if "error" in st.session_state:
        title, message, hint = st.session_state["error"]
        st.error(f"**{title}**\n\n{_esc(message)}" + (f"\n\n{hint}" if hint else ""))
    if "result" in st.session_state:
        render_result(st.session_state["result"])
