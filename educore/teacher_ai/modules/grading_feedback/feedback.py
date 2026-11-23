
from .schemas import GradingResult


def generate_student_feedback(result: GradingResult) -> dict:
    """
    Prepare a student-friendly feedback summary.

    This function does not modify or recalculate any grades.
    """

    if not isinstance(result, GradingResult):
        result = GradingResult.model_validate(result)

    question_feedback = []

    for grade in result.question_grades:
        question_feedback.append({
            "question_id": grade.question_id,
            "awarded_marks": grade.awarded_marks,
            "max_marks": grade.max_marks,
            "is_fully_correct": grade.is_fully_correct,
            "feedback": grade.feedback,
            "strengths": list(grade.strengths),
            "improvements": list(grade.improvements),
            "criterion_feedback": [
                {
                    "criterion_id": criterion.criterion_id,
                    "awarded_marks": criterion.awarded_marks,
                    "feedback": criterion.feedback,
                }
                for criterion in grade.criterion_grades
            ],
        })

    return {
        "total_awarded_marks": result.total_awarded_marks,
        "total_max_marks": result.total_max_marks,
        "percentage": result.percentage,
        "overall_feedback": result.overall_feedback,
        "question_feedback": question_feedback,
    }


def get_performance_summary(result: GradingResult) -> str:
    """Create a concise textual summary of student performance."""

    if not isinstance(result, GradingResult):
        result = GradingResult.model_validate(result)

    percentage = result.percentage

    if percentage >= 90:
        performance = "Excellent performance"
    elif percentage >= 75:
        performance = "Very good performance"
    elif percentage >= 60:
        performance = "Good effort"
    elif percentage >= 50:
        performance = "Needs improvement"
    else:
        performance = "Requires additional practice"

    return (
        f"{performance}: "
        f"{result.total_awarded_marks:g}/"
        f"{result.total_max_marks:g} marks "
        f"({percentage:.1f}%). "
        f"{result.overall_feedback}"
    )
