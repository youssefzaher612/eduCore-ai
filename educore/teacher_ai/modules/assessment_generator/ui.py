
"""EduCore Teacher AI - Assessment Generator UI."""

import json
import logging

import streamlit as st
from pydantic import ValidationError

from educore.teacher_ai.core.llm_client import LLMError, LLMRateLimitError, LLMRequestError

from .generator import generate_assessment
from .generator import AssessmentGenerationError
from .presentation import (
    format_assessment_text,
    get_assessment_summary,
    get_student_version,
    get_teacher_version,
)
from .schemas import AssessmentRequest

logger = logging.getLogger("educore.teacher.assessment_ui")


def _missing_required_fields(subject: str | None, grade: str | None) -> list[str]:
    """Return required assessment labels that contain no non-whitespace text."""
    missing = []
    if not str(subject or "").strip():
        missing.append("subject")
    if not str(grade or "").strip():
        missing.append("grade")
    return missing


def _build_assessment_request(subject: str, grade: str, **fields):
    """Build a validated request or return a user-facing form error."""
    missing = _missing_required_fields(subject, grade)
    if missing:
        return None, "Please enter both a subject and a grade before generating the assessment."
    try:
        return AssessmentRequest(
            subject=subject.strip(),
            grade=grade.strip(),
            **fields,
        ), None
    except ValidationError:
        return None, "Some assessment details are incomplete or invalid. Check the form and try again."


def render_assessment_page(
    source_material: str = "",
    selected_topic: str = "",
    source_file: str = "",
) -> None:
    """Render the assessment generation interface."""

    st.title("📝 Assessment Generator")
    st.caption("Create quizzes, homework and exams from the selected indexed course file.")
    if not source_material.strip() or not selected_topic:
        st.info("Choose an indexed course file and one of its topics above to create a RAG-grounded assessment.")
        return
    st.info(f"Assessment topic: **{selected_topic}** · RAG source: **{source_file}**")

    with st.form("assessment_generator_form"):
        st.subheader("Assessment details")

        c1, c2 = st.columns(2)
        subject = c1.text_input(
            "Subject", placeholder="e.g. Mathematics"
        )
        grade = c2.text_input(
            "Grade / level", placeholder="e.g. Grade 9"
        )

        topic = st.text_input(
            "Topic from the selected course file",
            value=selected_topic,
            disabled=True,
        )

        c3, c4 = st.columns(2)
        assessment_type = c3.selectbox(
            "Assessment type",
            ["quiz", "homework", "exam"],
        )
        difficulty = c4.selectbox(
            "Difficulty",
            ["easy", "medium", "hard", "mixed"],
        )

        question_types = st.multiselect(
            "Question types",
            [
                "mcq",
                "true_false",
                "short_answer",
                "problem_solving",
                "essay",
            ],
            default=["mcq", "short_answer"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        c5, c6, c7 = st.columns(3)
        question_count = c5.number_input(
            "Number of questions",
            min_value=1,
            max_value=20,
            value=4,
        )
        total_marks = c6.number_input(
            "Total marks",
            min_value=1.0,
            max_value=1000.0,
            value=20.0,
            step=1.0,
        )
        duration = c7.number_input(
            "Duration (minutes)",
            min_value=1,
            max_value=300,
            value=30,
        )

        language = st.selectbox(
            "Assessment language",
            ["English", "Arabic"],
        )

        additional_instructions = st.text_area(
            "Additional instructions (optional)",
            placeholder=(
                "Mention curriculum requirements, specific skills, "
                "or topics to focus on..."
            ),
        )

        submitted = st.form_submit_button(
            "Generate Assessment",
            type="primary",
            width="stretch",
        )

    if submitted:
        st.session_state.pop("assessment_result", None)
        st.session_state.pop("assessment_error", None)

        request, validation_message = _build_assessment_request(
            subject,
            grade,
            topic=topic,
            assessment_type=assessment_type,
            difficulty=difficulty,
            question_types=question_types,
            question_count=int(question_count),
            total_marks=float(total_marks),
            duration_minutes=int(duration),
            language=language,
            additional_instructions=additional_instructions,
        )
        if validation_message:
            st.warning(validation_message)
            return

        try:
            with st.status(
                "Generating your assessment...",
                expanded=True,
            ) as status:
                st.write("Creating questions and model answers...")
                result = generate_assessment(
                    request,
                    source_material=source_material,
                )
                status.update(
                    label="Assessment generated successfully",
                    state="complete",
                    expanded=False,
                )

            st.session_state["assessment_result"] = result

        except LLMRateLimitError as exc:
            wait = f" انتظر حوالي {max(1, int(exc.retry_after))} ثانية وجرب تاني." if exc.retry_after else " انتظر قليلًا وجرب تاني."
            st.session_state["assessment_error"] = "تم تجاوز حد الطلبات أو التوكنات المؤقت لدى Groq (TPM/RPM)." + wait
        except LLMRequestError as exc:
            if exc.status_code == 413:
                st.session_state["assessment_error"] = (
                    "الطلب أكبر من حد المزوّد حتى بعد تقليصه. قلل عدد الأسئلة أو التعليمات، "
                    "ثم أعد تشغيل الـAPI بعد تحديث الإعدادات."
                )
            else:
                st.session_state["assessment_error"] = exc.message
        except (LLMError, AssessmentGenerationError) as exc:
            st.session_state["assessment_error"] = exc.message if isinstance(exc, LLMError) else str(exc)
        except Exception:
            logger.warning("AI generation task=teacher_exam stage=failed error_category=internal")
            st.session_state["assessment_error"] = (
                "تعذر إنشاء الامتحان بسبب خطأ داخلي. التفاصيل محفوظة في سجل تشغيل الواجهة."
            )

    if "assessment_error" in st.session_state:
        st.error(st.session_state["assessment_error"])

    if "assessment_result" in st.session_state:
        _render_assessment_result(
            st.session_state["assessment_result"]
        )


def _render_assessment_result(assessment) -> None:
    """Render generated assessment versions and downloads."""

    summary = get_assessment_summary(assessment)

    st.divider()
    st.subheader("Assessment overview")

    st.success("Your assessment is ready.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Questions", summary["question_count"])
    c2.metric("Total marks", summary["total_marks"])
    c3.metric("Duration", f'{summary["duration_minutes"]} min')

    with st.expander("Assessment distribution"):
        st.write("**Question types**")
        st.json(summary["question_types"])

        st.write("**Difficulty distribution**")
        st.json(summary["difficulty_distribution"])

    student = get_student_version(assessment)
    teacher = assessment.teacher_version

    tab_student, tab_teacher, tab_download = st.tabs(
        ["Student Version", "Teacher Version", "Downloads"]
    )

    with tab_student:
        st.subheader(student["title"])

        st.write(f"**Total marks:** {student['total_marks']}")
        st.write(f"**Duration:** {student['duration_minutes']} minutes")

        for instruction in student["instructions"]:
            st.markdown(f"- {instruction}")

        for question in student["questions"]:
            with st.container(border=True):
                st.markdown(
                    f"**Question {question['question_number']}** "
                    f"({question['marks']} marks)"
                )
                st.write(question["question_text"])

                for option in question["options"]:
                    st.write(f"**{option['id']}.** {option['text']}")

    with tab_teacher:
        st.subheader(teacher.title)
        st.caption("Teacher-only content: answers and marking guidance.")

        for question in teacher.questions:
            with st.expander(
                f"Question {question.question_number} "
                f"— {question.marks} marks"
            ):
                st.write(question.question_text)

                for option in question.options:
                    st.write(f"{option.id}. {option.text}")

                teacher_answer = question.model_answer or question.expected_answer
                if teacher_answer is None:
                    teacher_answer = question.correct_answer
                if question.question_type.value == "essay":
                    st.markdown("**Model answer**")
                    st.write(teacher_answer if teacher_answer is not None else "Not provided")
                else:
                    st.success(
                        f"Correct answer: {teacher_answer if teacher_answer is not None else 'Not provided'}"
                    )

                if question.explanation:
                    st.markdown("**Explanation**")
                    st.write(question.explanation)

                rubric = question.grading_rubric or question.marking_scheme
                if rubric:
                    st.markdown("**Marking scheme**")
                    for point in rubric:
                        st.write(
                            f"- {point.description}: "
                            f"{point.marks} marks"
                        )

    with tab_download:
        st.subheader("Download assessment")

        student_text = format_assessment_text(
            assessment.student_version,
            include_answers=False,
        )
        teacher_text = format_assessment_text(
            teacher,
            include_answers=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Student Version**")
            st.caption("Questions and instructions only.")

            st.download_button(
                "Download Student Version",
                data=student_text,
                file_name="student_assessment.txt",
                mime="text/plain",
                width="stretch",
            )

        with c2:
            st.markdown("**Teacher Version**")
            st.caption("Questions, answers and marking schemes.")

            st.download_button(
                "Download Teacher Version",
                data=teacher_text,
                file_name="teacher_assessment.txt",
                mime="text/plain",
                width="stretch",
            )

        st.download_button(
            "Download Student Assessment (JSON)",
            data=json.dumps(
                student,
                ensure_ascii=False,
                indent=2,
            ),
            file_name="student_assessment.json",
            mime="application/json",
            width="stretch",
        )
        st.download_button(
            "Download Teacher Assessment (JSON)",
            data=json.dumps(
                get_teacher_version(assessment),
                ensure_ascii=False,
                indent=2,
            ),
            file_name="teacher_assessment.json",
            mime="application/json",
            width="stretch",
        )
