"""Streamlit interface for reports and teacher communication."""

import json
import logging

import streamlit as st

from .communication_generator import generate_communication
from .presentation import (
    get_class_report_data,
    get_communication_data,
    get_student_report_data,
)
from .report_generator import generate_class_report, generate_student_report
from .schemas import CommunicationRequest, ReportRequest

logger = logging.getLogger("educore.teacher.reports_ui")


def render_reports_page(
    source_material: str = "",
    source_file: str = "",
) -> None:
    """Render reports from saved analytics and communication generation."""

    st.title("📨 Reports & Communication")
    st.caption(
        "Prepare clear performance reports and supportive messages for students and families."
    )
    analytics_result = st.session_state.get("analytics_result")

    if analytics_result is None:
        st.info(
            "Run Student Performance Analytics first to create an individual or class report. "
            "You can still draft a communication message below."
        )
    else:
        _render_report_forms(analytics_result)

    st.divider()
    _render_communication_form(analytics_result, source_material, source_file)

    _render_saved_outputs()


def _render_report_forms(analytics_result) -> None:
    student_tab, class_tab = st.tabs(["Student report", "Class report"])
    subject = analytics_result.subject
    grade_level = analytics_result.grade_level

    with student_tab:
        st.subheader("Individual student report")
        student_options = {
            student.student_id: student.student_name
            for student in analytics_result.students
        }
        with st.form("student_report_form"):
            report_title = st.text_input(
                "Report title",
                value=f"{subject} student progress report",
                key="student_report_title",
            )
            selected_id = st.selectbox(
                "Student",
                list(student_options),
                format_func=lambda student_id: (
                    f"{student_options[student_id]} ({student_id})"
                ),
            )
            include_recommendations = st.checkbox(
                "Include recommendations and next steps",
                value=True,
                key="student_report_recommendations",
            )
            submitted = st.form_submit_button(
                "Generate student report",
                type="primary",
                width="stretch",
            )
        if submitted:
            st.session_state.pop("student_report_result", None)
            st.session_state.pop("student_report_error", None)
            request = ReportRequest(
                report_title=report_title,
                subject=subject,
                grade_level=grade_level,
                student_id=selected_id,
                student_name=student_options[selected_id],
                include_recommendations=include_recommendations,
            )
            try:
                st.session_state["student_report_result"] = generate_student_report(
                    request,
                    analytics_result,
                )
            except Exception:
                logger.exception("Student report generation failed")
                st.session_state["student_report_error"] = (
                    "The student report could not be generated. Check the selected analytics "
                    "result and try again."
                )

    with class_tab:
        st.subheader("Class performance report")
        with st.form("class_report_form"):
            report_title = st.text_input(
                "Report title",
                value=f"{subject} class performance report",
                key="class_report_title",
            )
            include_recommendations = st.checkbox(
                "Include recommendations and next steps",
                value=True,
                key="class_report_recommendations",
            )
            submitted = st.form_submit_button(
                "Generate class report",
                type="primary",
                width="stretch",
            )
        if submitted:
            st.session_state.pop("class_report_result", None)
            st.session_state.pop("class_report_error", None)
            request = ReportRequest(
                report_title=report_title,
                subject=subject,
                grade_level=grade_level,
                include_recommendations=include_recommendations,
            )
            try:
                st.session_state["class_report_result"] = generate_class_report(
                    request,
                    analytics_result,
                )
            except Exception:
                logger.exception("Class report generation failed")
                st.session_state["class_report_error"] = (
                    "The class report could not be generated. Review the analytics results "
                    "and try again."
                )


def _render_communication_form(
    analytics_result,
    source_material: str = "",
    source_file: str = "",
) -> None:
    st.subheader("Parent or student communication")
    student_options = {"": "Enter a student name"}
    if analytics_result is not None:
        student_options.update(
            {
                student.student_id: student.student_name
                for student in analytics_result.students
            }
        )
    with st.form("communication_form"):
        recipient_type = st.selectbox(
            "Recipient",
            ["Parent / guardian", "Student", "School staff"],
        )
        recipient_name = st.text_input(
            "Recipient name",
            placeholder="e.g. Taylor Morgan",
        )
        selected_student_id = st.selectbox(
            "Link a student (optional)",
            list(student_options),
            format_func=lambda student_id: student_options[student_id],
        )
        linked_student_name = student_options[selected_student_id]
        student_name = st.text_input(
            "Student name",
            value=linked_student_name if selected_student_id else "",
            placeholder="e.g. Alex Morgan",
        )
        c1, c2, c3 = st.columns(3)
        subject = c1.text_input(
            "Subject",
            value=analytics_result.subject if analytics_result is not None else "",
        )
        grade_level = c2.text_input(
            "Grade / level",
            value=analytics_result.grade_level if analytics_result is not None else "",
        )
        language = c3.selectbox("Message language", ["English", "Arabic"])
        purpose = st.text_area(
            "Purpose",
            placeholder="e.g. Share progress and agree on a revision plan",
        )
        tone = st.selectbox(
            "Tone",
            ["professional", "warm", "encouraging", "formal"],
            format_func=lambda value: value.title(),
        )
        additional_context = st.text_area(
            "Additional context (optional)",
            placeholder=(
                f"The selected RAG file ({source_file}) will be included as course context. "
                "Add any student-specific details you want reflected in the message."
                if source_file else "Include only details you want reflected in the message"
            ),
        )
        submitted = st.form_submit_button(
            "Generate communication",
            type="primary",
            width="stretch",
        )

    if submitted:
        st.session_state.pop("communication_result", None)
        st.session_state.pop("communication_error", None)
        missing_fields = []
        if not recipient_name.strip():
            missing_fields.append("recipient name")
        if not student_name.strip():
            missing_fields.append("student name")
        if not subject.strip():
            missing_fields.append("subject")
        if not grade_level.strip():
            missing_fields.append("grade / level")
        if missing_fields:
            st.session_state["communication_error"] = (
                "Please complete the required fields: " + ", ".join(missing_fields)
            )
        elif not purpose.strip():
            st.session_state["communication_error"] = "Enter the purpose of the message."
        else:
            request = CommunicationRequest(
                recipient_type=recipient_type,
                recipient_name=recipient_name,
                student_name=student_name,
                subject=subject,
                grade_level=grade_level,
                purpose=f"{purpose.strip()}\nRequested language: {language}.",
                tone=tone,
                additional_context="\n\n".join(part for part in (
                    additional_context.strip(),
                    (
                        "Selected course-file excerpts (use only if relevant; do not invent student performance):\n"
                        + source_material[:6000]
                    ) if source_material.strip() else "",
                ) if part) or None,
            )
            try:
                with st.status("Preparing your message…", expanded=True) as status:
                    st.write("Drafting a respectful, supportive communication…")
                    result = generate_communication(request)
                    status.update(
                        label="Message ready",
                        state="complete",
                        expanded=False,
                    )
                st.session_state["communication_result"] = result
            except Exception:
                logger.exception("Communication generation failed")
                st.session_state["communication_error"] = (
                    "The message could not be generated. Review the details and try again."
                )


def _render_saved_outputs() -> None:
    for key, title in (
        ("student_report_error", "Student report"),
        ("class_report_error", "Class report"),
        ("communication_error", "Communication"),
    ):
        if key in st.session_state:
            st.error(f"{title}: {st.session_state[key]}")

    student_report = st.session_state.get("student_report_result")
    if student_report is not None:
        data = get_student_report_data(student_report)
        st.markdown("#### Student report")
        st.success(
            f"{student_report.student_name}: "
            f"{student_report.average_percentage:.1f}% · "
            f"{student_report.performance_status}"
        )
        _render_report_sections(student_report.sections, student_report.recommendations)
        st.download_button(
            "Download student report (JSON)",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name="student_progress_report.json",
            mime="application/json",
            key="download_student_report",
        )

    class_report = st.session_state.get("class_report_result")
    if class_report is not None:
        data = get_class_report_data(class_report)
        st.markdown("#### Class report")
        st.success(
            f"Class average: {class_report.class_average:.1f}% · "
            f"Pass rate: {class_report.pass_rate:.1f}%"
        )
        _render_report_sections(class_report.sections, class_report.recommendations)
        st.download_button(
            "Download class report (JSON)",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name="class_performance_report.json",
            mime="application/json",
            key="download_class_report",
        )

    communication = st.session_state.get("communication_result")
    if communication is not None:
        data = get_communication_data(communication)
        st.markdown("#### Generated message")
        st.success(f"**{communication.subject_line}**")
        st.caption(
            f"To: {communication.recipient_name} · "
            f"{communication.tone.title()} · {communication.language}"
        )
        st.markdown(communication.message)
        st.download_button(
            "Download message (text)",
            data=f"Subject: {communication.subject_line}\n\n{communication.message}",
            file_name="teacher_communication.txt",
            mime="text/plain",
            key="download_communication_text",
        )
        st.download_button(
            "Download message (JSON)",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name="teacher_communication.json",
            mime="application/json",
            key="download_communication_json",
        )


def _render_report_sections(sections, recommendations) -> None:
    for section in sections:
        with st.expander(section.heading, expanded=True):
            st.write(section.content)
    if recommendations:
        st.markdown("**Recommendations and next steps**")
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")
