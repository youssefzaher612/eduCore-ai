"""Embedded teacher workspace using EduCore's indexed files and student records."""

from __future__ import annotations

import streamlit as st

from educore.teacher_ai.lesson_ui import render_lesson_page
from educore.teacher_ai.modules.assessment_generator.ui import render_assessment_page
from educore.teacher_ai.modules.grading_feedback.ui import render_grading_page
from educore.teacher_ai.modules.reports_communication.ui import render_reports_page
from educore.teacher_ai.modules.student_analytics.ui import render_analytics_records


def render_teacher_workspace(user_id: str, project_id: str, api_get) -> None:
    st.subheader("🧑‍🏫 أدوات المدرّس")
    st.caption(
        "تحضير الدروس والامتحانات والتصحيح مبني على ملفات المقرر المفهرسة، "
        "والتحليلات والتقارير مبنية على محاولات الامتحان المسجلة."
    )

    documents = api_get("/documents", {"project_id": project_id}) or []
    files = [
        item for item in documents
        if item.get("status") == "indexed" and int(item.get("total_chunks") or 0) > 0
    ]
    if not files:
        st.info("ارفع ملف المقرر من مستودع الملفات وانتظر اكتمال الفهرسة، وبعدها هتظهر أدوات المدرّس هنا.")
        return

    file_ids = [item["id"] for item in files]
    if st.session_state.get("teacher_source_document_id") not in file_ids:
        st.session_state.teacher_source_document_id = file_ids[0]
    selected_document_id = st.selectbox(
        "ملف المقرر من الـRAG",
        options=file_ids,
        format_func=lambda value: next(
            (item["filename"] for item in files if item["id"] == value), value
        ),
        key="teacher_source_document_id",
    )
    selected_file = next(item for item in files if item["id"] == selected_document_id)

    dashboard = api_get(
        "/student/dashboard",
        {
            "user_id": user_id,
            "project_id": project_id,
            "document_id": selected_document_id,
        },
    ) or {}
    topics = [
        str(item.get("name", "")).strip()
        for item in dashboard.get("course_topics", [])
        if str(item.get("name", "")).strip()
    ]
    if st.session_state.get("teacher_source_topic") not in topics:
        st.session_state.teacher_source_topic = topics[0] if topics else None
    selected_topic = st.selectbox(
        "موضوع العمل من الملف",
        options=topics,
        key="teacher_source_topic",
        disabled=not topics,
        placeholder="لا توجد موضوعات مفهرسة في هذا الملف",
    )

    current_source = (selected_document_id, selected_topic)
    previous_source = st.session_state.get("teacher_previous_source")
    if previous_source is not None and previous_source != current_source:
        for key in (
            "result", "error",
            "assessment_result", "assessment_error",
            "grading_result", "grading_request", "grading_error",
            "analytics_result", "analytics_error",
            "student_report_result", "student_report_error",
            "class_report_result", "class_report_error",
            "communication_result", "communication_error",
        ):
            st.session_state.pop(key, None)
    st.session_state.teacher_previous_source = current_source

    if not topics or not selected_topic:
        st.warning("الملف مفهرس لكن مفيش موضوعات ظاهرة. أعد فهرسته من مستودع الملفات.")
        return

    material = api_get(
        "/teacher/materials",
        {
            "user_id": user_id,
            "project_id": project_id,
            "document_id": selected_document_id,
            "topic": selected_topic,
        },
    )
    if not material or not material.get("context"):
        st.warning("مقدرتش أجيب مقاطع مناسبة من الملف المختار للموضوع ده.")
        return

    st.caption(f"المصدر الحالي: **{selected_file['filename']}** · **{selected_topic}**")
    with st.expander("معاينة المقاطع المسترجعة من الملف"):
        for excerpt in material.get("excerpts", []):
            page = excerpt.get("page_number")
            st.markdown(
                f"**{excerpt['document_name']}**"
                + (f" · صفحة {page}" if page else "")
            )
            st.write(excerpt.get("content", ""))

    source_material = material["context"]
    lesson_tab, assessment_tab, grading_tab, analytics_tab, reports_tab = st.tabs(
        [
            "📘 تحضير الدرس",
            "📝 إنشاء امتحان",
            "✅ التصحيح والإفادة",
            "📊 أداء الطلاب",
            "📨 التقارير والتواصل",
        ]
    )

    with lesson_tab:
        render_lesson_page(source_material, selected_topic, selected_file["filename"])

    with assessment_tab:
        render_assessment_page(source_material, selected_topic, selected_file["filename"])

    with grading_tab:
        render_grading_page(source_material, selected_topic, selected_file["filename"])

    with analytics_tab:
        records_result = api_get(
            "/teacher/assessment-records",
            {
                "user_id": user_id,
                "project_id": project_id,
                "document_id": selected_document_id,
            },
        ) or {}
        analytics_result = render_analytics_records(
            records_result.get("records", []),
            subject=records_result.get("project_name") or selected_file["filename"],
            grade_level="غير محدد",
        )
        if analytics_result is not None:
            st.caption("المصدر: درجات الامتحانات المسلّمة المسجلة في EduCore، مش ملف بيانات خارجي.")

    with reports_tab:
        render_reports_page(source_material, selected_file["filename"])
