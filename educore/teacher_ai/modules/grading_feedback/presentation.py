
from .schemas import GradingResult


def get_student_version(result: GradingResult) -> dict:
    """Return a student-friendly version of the grading result."""

    if not isinstance(result, GradingResult):
        result = GradingResult.model_validate(result)

    return {
        "total_awarded_marks": result.total_awarded_marks,
        "total_max_marks": result.total_max_marks,
        "percentage": result.percentage,
        "overall_feedback": result.overall_feedback,
        "questions": [
            {
                "question_id": grade.question_id,
                "awarded_marks": grade.awarded_marks,
                "max_marks": grade.max_marks,
                "is_fully_correct": grade.is_fully_correct,
                "feedback": grade.feedback,
                "strengths": grade.strengths,
                "improvements": grade.improvements,
            }
            for grade in result.question_grades
        ],
    }


def get_teacher_version(result: GradingResult) -> dict:
    """Return a detailed grading report for the teacher."""

    if not isinstance(result, GradingResult):
        result = GradingResult.model_validate(result)

    return {
        "total_awarded_marks": result.total_awarded_marks,
        "total_max_marks": result.total_max_marks,
        "percentage": result.percentage,
        "overall_feedback": result.overall_feedback,
        "question_grades": [
            grade.model_dump(mode="json")
            for grade in result.question_grades
        ],
    }


def format_grading_text(result: GradingResult) -> str:
    """Format grading results as readable text."""

    if not isinstance(result, GradingResult):
        result = GradingResult.model_validate(result)

    lines = [
        "EduCore - Grading Report",
        "=" * 35,
        (
            f"Total: {result.total_awarded_marks:g}/"
            f"{result.total_max_marks:g}"
        ),
        f"Percentage: {result.percentage:.1f}%",
        "",
        "Overall Feedback:",
        result.overall_feedback,
        "",
        "Question Breakdown:",
    ]

    for grade in result.question_grades:
        status = "Correct" if grade.is_fully_correct else "Needs Review"

        lines.extend([
            "",
            f"{grade.question_id} [{status}]",
            (
                f"Marks: {grade.awarded_marks:g}/"
                f"{grade.max_marks:g}"
            ),
            f"Feedback: {grade.feedback}",
        ])

        if grade.strengths:
            lines.append("Strengths:")
            lines.extend(f"- {item}" for item in grade.strengths)

        if grade.improvements:
            lines.append("Areas to Improve:")
            lines.extend(f"- {item}" for item in grade.improvements)

    return "\n".join(lines)
