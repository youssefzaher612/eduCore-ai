
from .schemas import AssessmentResult, GeneratedAssessment


def get_student_version(
    assessment: GeneratedAssessment,
) -> dict:
    """Return only student-facing assessment content."""

    result = assessment.student_version.model_dump()

    for question in result["questions"]:
        question.pop("correct_answer", None)
        question.pop("explanation", None)
        question.pop("marking_scheme", None)
        question.pop("model_answer", None)
        question.pop("grading_rubric", None)
        question.pop("expected_answer", None)
        question.pop("acceptable_alternatives", None)

    return result


def get_teacher_version(
    assessment: GeneratedAssessment,
) -> dict:
    """Return the teacher version with answers and marking schemes."""

    return assessment.teacher_version.model_dump()


def get_assessment_summary(
    assessment: GeneratedAssessment,
) -> dict:
    """Build a compact assessment overview."""

    student = assessment.student_version
    questions = student.questions

    type_counts = {}

    for question in questions:
        question_type = question.question_type.value
        type_counts[question_type] = (
            type_counts.get(question_type, 0) + 1
        )

    difficulty_counts = {}

    for question in questions:
        difficulty = question.difficulty.value
        difficulty_counts[difficulty] = (
            difficulty_counts.get(difficulty, 0) + 1
        )

    return {
        "title": student.title,
        "question_count": len(questions),
        "total_marks": student.total_marks,
        "duration_minutes": student.duration_minutes,
        "question_types": type_counts,
        "difficulty_distribution": difficulty_counts,
    }


def format_assessment_text(
    result: AssessmentResult,
    include_answers: bool = False,
) -> str:
    """Format an assessment as readable plain text."""

    lines = [
        result.title,
        "",
        f"Total Marks: {result.total_marks}",
        f"Duration: {result.duration_minutes} minutes",
        "",
        "Instructions:",
    ]

    for instruction in result.instructions:
        lines.append(f"- {instruction}")

    lines.append("")

    for question in result.questions:
        lines.append(
            f"Q{question.question_number}. "
            f"{question.question_text}"
        )

        lines.append(f"Marks: {question.marks}")

        for option in question.options:
            lines.append(f"   {option.id}) {option.text}")

        if include_answers:
            answer = question.model_answer or question.expected_answer
            if answer is None:
                answer = question.correct_answer
            lines.append(f"Answer: {answer if answer is not None else 'Not provided'}")

            if question.explanation:
                lines.append(f"Explanation: {question.explanation}")

            rubric = question.grading_rubric or question.marking_scheme
            for point in rubric:
                lines.append(
                    f"- {point.description}: {point.marks} marks"
                )

        lines.append("")

    return "\n".join(lines)
