"""Streamlit interface for grading and constructive feedback."""

import json
import logging

import streamlit as st
from pydantic import ValidationError

from .grader import grade_submission
from .presentation import format_grading_text, get_teacher_version
from .schemas import (
    GradingQuestion,
    GradingRequest,
    MarkingCriterion,
    StudentAnswer,
)

logger = logging.getLogger("educore.teacher.grading_ui")


def render_grading_page(
    source_material: str = "",
    selected_topic: str = "",
    source_file: str = "",
) -> None:
    """Render the assessment grading form and its saved result."""

    st.title("✅ Grading & Feedback")
    st.caption("Apply clear marking criteria and prepare supportive feedback, checked against the selected course file.")
    if not source_material.strip() or not selected_topic:
        st.info("Choose an indexed course file and one of its topics above to ground grading in the course material.")
        return
    st.info(f"Grading context: **{selected_topic}** · RAG source: **{source_file}**")

    question_count = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
        key="grading_question_count",
        help="Add the questions and student answers below.",
    )

    with st.form("grading_feedback_form"):
        st.subheader("Assessment details")
        c1, c2 = st.columns(2)
        subject = c1.text_input("Subject", placeholder="e.g. Mathematics")
        grade_level = c2.text_input("Grade / level", placeholder="e.g. Grade 9")
        c3, c4 = st.columns(2)
        language = c3.selectbox("Feedback language", ["English", "Arabic"])
        teacher_instructions = c4.text_input(
            "Teacher instructions (optional)",
            placeholder="e.g. Be lenient with equivalent methods",
        )

        entries = []
        st.subheader("Questions and student answers")
        st.caption(
            "Enter marking criteria one per line. Criteria share the question's "
            "maximum marks equally."
        )
        for index in range(int(question_count)):
            question_id = f"Q{index + 1}"
            with st.container(border=True):
                st.markdown(f"**Question {index + 1}**")
                question_text = st.text_area(
                    "Question",
                    key=f"grading_question_text_{index}",
                    height=70,
                    placeholder="Enter the question shown to the student",
                )
                col_type, col_marks = st.columns(2)
                question_type = col_type.selectbox(
                    "Question type",
                    ["mcq", "true_false", "short_answer", "problem_solving", "essay"],
                    format_func=lambda value: value.replace("_", " ").title(),
                    key=f"grading_question_type_{index}",
                )
                max_marks = col_marks.number_input(
                    "Maximum marks",
                    min_value=0.5,
                    max_value=1000.0,
                    value=5.0,
                    step=0.5,
                    key=f"grading_max_marks_{index}",
                )
                expected_answer = st.text_area(
                    "Expected answer / model answer",
                    key=f"grading_expected_answer_{index}",
                    height=65,
                )
                criteria_text = st.text_area(
                    "Marking criteria (optional)",
                    key=f"grading_criteria_{index}",
                    height=65,
                    placeholder="One criterion per line",
                )
                student_response = st.text_area(
                    "Student answer",
                    key=f"grading_student_answer_{index}",
                    height=80,
                )
                entries.append(
                    {
                        "question_id": question_id,
                        "question_text": question_text,
                        "question_type": question_type,
                        "max_marks": float(max_marks),
                        "expected_answer": expected_answer,
                        "criteria_text": criteria_text,
                        "student_response": student_response,
                    }
                )

        submitted = st.form_submit_button(
            "Grade submission",
            type="primary",
            width="stretch",
        )

    if submitted:
        st.session_state.pop("grading_result", None)
        st.session_state.pop("grading_request", None)
        st.session_state.pop("grading_error", None)
        questions = []
        answers = []
        for entry in entries:
            criteria = [
                line.strip()
                for line in entry["criteria_text"].splitlines()
                if line.strip()
            ]
            criterion_marks = entry["max_marks"] / len(criteria) if criteria else 0
            questions.append(
                GradingQuestion(
                    question_id=entry["question_id"],
                    question_text=entry["question_text"],
                    question_type=entry["question_type"],
                    max_marks=entry["max_marks"],
                    expected_answer=entry["expected_answer"] or None,
                    marking_criteria=[
                        MarkingCriterion(
                            criterion_id=f"C{criterion_index}",
                            description=description,
                            max_marks=criterion_marks,
                        )
                        for criterion_index, description in enumerate(criteria, start=1)
                    ],
                )
            )
            if entry["student_response"].strip():
                answers.append(
                    StudentAnswer(
                        question_id=entry["question_id"],
                        answer=entry["student_response"].strip(),
                    )
                )

        missing_fields = []
        if not subject.strip():
            missing_fields.append("subject")
        if not grade_level.strip():
            missing_fields.append("grade / level")
        for index, entry in enumerate(entries, start=1):
            if not entry["question_text"].strip():
                missing_fields.append(f"question {index}")

        if missing_fields:
            st.session_state["grading_error"] = (
                "Please complete the required fields: " + ", ".join(missing_fields)
            )
        elif not answers:
            st.session_state["grading_error"] = (
                "Enter at least one student answer before grading."
            )
        else:
            try:
                request = GradingRequest(
                    subject=subject,
                    grade_level=grade_level,
                    language=language,
                    teacher_instructions=teacher_instructions or None,
                    course_material=source_material[:12000],
                    questions=questions,
                    student_answers=answers,
                )
            except ValidationError as exc:
                fields = [
                    ".".join(str(part) for part in error["loc"])
                    for error in exc.errors()
                ]
                st.session_state["grading_error"] = (
                    "Please complete the required grading fields: "
                    + ", ".join(fields)
                )
            else:
                try:
                    with st.status("Grading the submission…", expanded=True) as status:
                        st.write("Applying the supplied answers and marking criteria…")
                        result = grade_submission(request)
                        status.update(
                            label="Grading complete",
                            state="complete",
                            expanded=False,
                        )
                    st.session_state["grading_request"] = request
                    st.session_state["grading_result"] = result
                except Exception:
                    logger.exception("Submission grading failed")
                    st.session_state["grading_error"] = (
                        "Grading could not be completed. Review the assessment details "
                        "and try again."
                    )

    if "grading_error" in st.session_state:
        st.error(st.session_state["grading_error"])
    if "grading_result" in st.session_state:
        _render_grading_result(
            st.session_state["grading_result"],
            st.session_state["grading_request"],
        )


def _render_grading_result(result, request: GradingRequest) -> None:
    st.divider()
    st.subheader("Grading results")
    st.success("Submission graded. Review the marks and feedback before sharing.")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Awarded marks",
        f"{result.total_awarded_marks:g} / {result.total_max_marks:g}",
    )
    c2.metric("Percentage", f"{result.percentage:.1f}%")
    c3.metric("Questions graded", len(result.question_grades))
    st.info(result.overall_feedback)

    questions = {question.question_id: question for question in request.questions}
    for grade in result.question_grades:
        question = questions[grade.question_id]
        with st.expander(
            f"{grade.question_id} · {grade.awarded_marks:g}/{grade.max_marks:g} marks"
        ):
            st.markdown(f"**Question:** {question.question_text}")
            st.write(grade.feedback)
            left, right = st.columns(2)
            with left:
                st.markdown("**Strengths**")
                if grade.strengths:
                    for item in grade.strengths:
                        st.markdown(f"- {item}")
                else:
                    st.caption("No strengths listed.")
            with right:
                st.markdown("**Areas for improvement**")
                if grade.improvements:
                    for item in grade.improvements:
                        st.markdown(f"- {item}")
                else:
                    st.caption("No improvement areas listed.")

            if grade.criterion_grades:
                criteria = {
                    item.criterion_id: item.description
                    for item in question.marking_criteria
                }
                st.markdown("**Criterion-level grading**")
                for criterion in grade.criterion_grades:
                    description = criteria.get(criterion.criterion_id, criterion.criterion_id)
                    st.write(
                        f"- {description}: {criterion.awarded_marks:g} marks — "
                        f"{criterion.feedback}"
                    )

    st.download_button(
        "Download grading report",
        data=format_grading_text(result),
        file_name="grading_report.txt",
        mime="text/plain",
        width="stretch",
    )
    st.download_button(
        "Download detailed grading data (JSON)",
        data=json.dumps(
            get_teacher_version(result),
            ensure_ascii=False,
            indent=2,
        ),
        file_name="grading_report.json",
        mime="application/json",
        width="stretch",
    )
