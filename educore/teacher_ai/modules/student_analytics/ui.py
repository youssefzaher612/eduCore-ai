"""Streamlit interface for deterministic student performance analytics."""

import json
import logging

import streamlit as st
from pydantic import ValidationError

from .analyzer import analyze_student_performance
from .insights import generate_student_insights
from .presentation import (
    get_analytics_report,
    get_analytics_summary,
    get_student_details,
    get_student_ranking,
    get_topic_analysis,
)
from .schemas import AnalyticsRequest, StudentAssessmentRecord, TopicPerformance

logger = logging.getLogger("educore.teacher.analytics_ui")


def render_analytics_page() -> None:
    """Render the performance-record entry form and saved analytics."""

    st.title("📊 Student Performance Analytics")
    st.caption(
        "Explore class outcomes, student progress and topic trends using deterministic analytics."
    )

    record_count = st.number_input(
        "Number of assessment records",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
        key="analytics_record_count",
        help="Add one row per student's assessment result.",
    )
    with st.form("student_analytics_form"):
        st.subheader("Class details")
        c1, c2, c3 = st.columns(3)
        subject = c1.text_input("Subject", placeholder="e.g. Mathematics")
        grade_level = c2.text_input("Grade / level", placeholder="e.g. Grade 9")
        pass_mark = c3.number_input(
            "Pass mark (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
        )

        st.subheader("Student assessment records")
        st.caption(
            "Use the same student ID for multiple assessments. Topic marks are optional "
            "and allow topic-level analysis."
        )
        entries = []
        for index in range(int(record_count)):
            with st.container(border=True):
                st.markdown(f"**Record {index + 1}**")
                identity1, identity2 = st.columns(2)
                student_id = identity1.text_input(
                    "Student ID",
                    key=f"analytics_student_id_{index}",
                    placeholder="e.g. S001",
                )
                student_name = identity2.text_input(
                    "Student name",
                    key=f"analytics_student_name_{index}",
                    placeholder="e.g. Alex Morgan",
                )
                details1, details2 = st.columns(2)
                assessment_title = details1.text_input(
                    "Assessment title",
                    key=f"analytics_assessment_title_{index}",
                    placeholder="e.g. Algebra quiz",
                )
                assessment_date = details2.text_input(
                    "Assessment date (optional)",
                    key=f"analytics_assessment_date_{index}",
                    placeholder="e.g. 2026-10-12",
                )
                marks1, marks2 = st.columns(2)
                awarded_marks = marks1.number_input(
                    "Marks awarded",
                    min_value=0.0,
                    max_value=1000.0,
                    value=15.0,
                    step=0.5,
                    key=f"analytics_awarded_marks_{index}",
                )
                max_marks = marks2.number_input(
                    "Maximum marks",
                    min_value=0.5,
                    max_value=1000.0,
                    value=20.0,
                    step=0.5,
                    key=f"analytics_max_marks_{index}",
                )
                topic1, topic2, topic3 = st.columns(3)
                topic_name = topic1.text_input(
                    "Topic (optional)",
                    key=f"analytics_topic_{index}",
                    placeholder="e.g. Linear equations",
                )
                topic_awarded = topic2.number_input(
                    "Topic marks",
                    min_value=0.0,
                    max_value=1000.0,
                    value=15.0,
                    step=0.5,
                    key=f"analytics_topic_awarded_{index}",
                )
                topic_max = topic3.number_input(
                    "Topic maximum",
                    min_value=0.5,
                    max_value=1000.0,
                    value=20.0,
                    step=0.5,
                    key=f"analytics_topic_max_{index}",
                )
                entries.append(
                    {
                        "student_id": student_id,
                        "student_name": student_name,
                        "assessment_title": assessment_title,
                        "assessment_date": assessment_date,
                        "awarded_marks": float(awarded_marks),
                        "max_marks": float(max_marks),
                        "topic": topic_name,
                        "topic_awarded": float(topic_awarded),
                        "topic_max": float(topic_max),
                    }
                )
        submitted = st.form_submit_button(
            "Analyze class performance",
            type="primary",
            width="stretch",
        )

    if submitted:
        st.session_state.pop("analytics_result", None)
        st.session_state.pop("analytics_error", None)
        try:
            missing_fields = []
            if not subject.strip():
                missing_fields.append("subject")
            if not grade_level.strip():
                missing_fields.append("grade / level")
            for index, entry in enumerate(entries, start=1):
                if not entry["student_id"].strip():
                    missing_fields.append(f"record {index} student ID")
                if not entry["student_name"].strip():
                    missing_fields.append(f"record {index} student name")
                if not entry["assessment_title"].strip():
                    missing_fields.append(f"record {index} assessment title")
                if entry["awarded_marks"] > entry["max_marks"]:
                    raise ValueError(
                        f"Record {index}: awarded marks cannot exceed maximum marks."
                    )
                if entry["topic"].strip() and entry["topic_awarded"] > entry["topic_max"]:
                    raise ValueError(
                        f"Record {index}: topic marks cannot exceed the topic maximum."
                    )
            if missing_fields:
                raise ValueError("Complete these fields: " + ", ".join(missing_fields))

            records = [
                StudentAssessmentRecord(
                    student_id=entry["student_id"],
                    student_name=entry["student_name"],
                    assessment_title=entry["assessment_title"],
                    subject=subject,
                    grade_level=grade_level,
                    awarded_marks=entry["awarded_marks"],
                    max_marks=entry["max_marks"],
                    assessment_date=entry["assessment_date"] or None,
                    topic_performance=(
                        [
                            TopicPerformance(
                                topic=entry["topic"],
                                awarded_marks=entry["topic_awarded"],
                                max_marks=entry["topic_max"],
                            )
                        ]
                        if entry["topic"].strip()
                        else []
                    ),
                )
                for entry in entries
            ]
            request = AnalyticsRequest(
                subject=subject,
                grade_level=grade_level,
                pass_mark=float(pass_mark),
                records=records,
            )
        except ValidationError as exc:
            fields = [
                ".".join(str(part) for part in error["loc"])
                for error in exc.errors()
            ]
            st.session_state["analytics_error"] = (
                "Please check the class and record fields: " + ", ".join(fields)
            )
        except ValueError as exc:
            st.session_state["analytics_error"] = str(exc)
        else:
            try:
                result = analyze_student_performance(request)
                result = generate_student_insights(result)
                st.session_state["analytics_result"] = result
            except Exception:
                logger.exception("Student performance analysis failed")
                st.session_state["analytics_error"] = (
                    "Analytics could not be calculated. Check that awarded marks do not "
                    "exceed the maximum marks, then try again."
                )

    if "analytics_error" in st.session_state:
        st.error(st.session_state["analytics_error"])
    if "analytics_result" in st.session_state:
        _render_analytics_result(st.session_state["analytics_result"])


def render_analytics_records(
    records: list[dict],
    subject: str,
    grade_level: str = "غير محدد",
    pass_mark: float = 50,
):
    """Render analytics calculated from EduCore's recorded exam attempts."""
    if not records:
        st.session_state.pop("analytics_result", None)
        st.info("مفيش امتحانات مسلّمة مسجلة للطلاب في الملف ده لسه.")
        return None
    try:
        parsed_records = [StudentAssessmentRecord.model_validate(item) for item in records]
        request = AnalyticsRequest(
            subject=subject,
            grade_level=grade_level,
            pass_mark=pass_mark,
            records=parsed_records,
        )
        result = generate_student_insights(analyze_student_performance(request))
    except Exception:
        logger.exception("EduCore assessment records could not be analyzed")
        st.error("تعذر حساب التحليلات من المحاولات المسجلة. راجع بيانات الامتحانات وحاول تاني.")
        return None
    st.session_state["analytics_result"] = result
    _render_analytics_result(result)
    return result


def _render_analytics_result(result) -> None:
    st.divider()
    st.subheader(f"{result.subject} · {result.grade_level}")
    st.success("Class analysis is ready.")
    summary = get_analytics_summary(result)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Class average", f"{summary['class_average']:.1f}%")
    m2.metric("Pass rate", f"{summary['pass_rate']:.1f}%")
    m3.metric("Students", summary["student_count"])
    m4.metric("Assessment records", summary["assessment_count"])

    rankings = get_student_ranking(result)
    if rankings:
        st.markdown("#### Student rankings")
        st.dataframe(rankings, hide_index=True, width="stretch")
        st.bar_chart(
            {
                "Student": [item["student_name"] for item in rankings],
                "Average (%)": [item["average_percentage"] for item in rankings],
            },
            x="Student",
            y="Average (%)",
            width="stretch",
        )

    topic_analysis = get_topic_analysis(result)
    if topic_analysis:
        st.markdown("#### Topic analysis")
        st.dataframe(topic_analysis, hide_index=True, width="stretch")
        st.bar_chart(
            {
                "Topic": [item["topic"] for item in topic_analysis],
                "Average (%)": [item["average_percentage"] for item in topic_analysis],
            },
            x="Topic",
            y="Average (%)",
            width="stretch",
        )
    else:
        st.info("Add topic marks to the records to see topic-level analysis.")

    students = get_student_details(result)
    if students:
        st.markdown("#### Individual student performance")
        selected_id = st.selectbox(
            "Select a student",
            [item["student_id"] for item in students],
            format_func=lambda student_id: next(
                item["student_name"]
                for item in students
                if item["student_id"] == student_id
            ),
            key="analytics_selected_student",
        )
        student = next(item for item in students if item["student_id"] == selected_id)
        c1, c2, c3 = st.columns(3)
        c1.metric("Average", f"{student['average_percentage']:.1f}%")
        c2.metric("Latest", f"{student['latest_percentage']:.1f}%")
        c3.metric("Assessments", student["assessment_count"])
        st.write(f"**Status:** {student['performance_status']}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Strengths**")
            for item in student["strengths"]:
                st.markdown(f"- {item}")
        with right:
            st.markdown("**Areas to support**")
            for item in student["weaknesses"]:
                st.markdown(f"- {item}")

    st.markdown("#### Recommendations")
    for recommendation in result.recommendations:
        st.markdown(f"- {recommendation}")

    st.download_button(
        "Download analytics report (JSON)",
        data=json.dumps(get_analytics_report(result), ensure_ascii=False, indent=2),
        file_name="student_analytics_report.json",
        mime="application/json",
        width="stretch",
    )
