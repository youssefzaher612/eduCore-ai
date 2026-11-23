
from .schemas import GradingRequest, GradingResult


TOLERANCE = 0.01


def validate_grading_result(
    request: GradingRequest,
    result: GradingResult,
) -> list[str]:
    """Validate grading consistency, IDs, and mark boundaries."""

    errors = []

    questions = {
        question.question_id: question
        for question in request.questions
    }

    answers = {
        answer.question_id: answer
        for answer in request.student_answers
    }

    grades = result.question_grades
    grade_ids = [grade.question_id for grade in grades]

    # Every submitted answer must have exactly one grade.
    if len(grade_ids) != len(set(grade_ids)):
        errors.append("Duplicate question IDs in grading results.")

    if set(grade_ids) != set(answers):
        errors.append(
            "Graded question IDs must exactly match submitted answer IDs."
        )

    expected_total_awarded = 0.0
    expected_total_max = 0.0

    for grade in grades:
        question = questions.get(grade.question_id)

        if question is None:
            errors.append(
                f"Unknown question ID: {grade.question_id}."
            )
            continue

        if grade.max_marks != question.max_marks:
            errors.append(
                f"{grade.question_id}: Maximum marks do not match the question."
            )

        if grade.awarded_marks > question.max_marks + TOLERANCE:
            errors.append(
                f"{grade.question_id}: Awarded marks exceed the question maximum."
            )

        if grade.awarded_marks < -TOLERANCE:
            errors.append(
                f"{grade.question_id}: Awarded marks cannot be negative."
            )

        expected_total_awarded += grade.awarded_marks
        expected_total_max += question.max_marks

        # Validate marking criteria.
        expected_criteria = {
            criterion.criterion_id: criterion
            for criterion in question.marking_criteria
        }

        actual_criteria_ids = [
            item.criterion_id for item in grade.criterion_grades
        ]

        if len(actual_criteria_ids) != len(set(actual_criteria_ids)):
            errors.append(
                f"{grade.question_id}: Duplicate criterion IDs."
            )

        if set(actual_criteria_ids) - set(expected_criteria):
            errors.append(
                f"{grade.question_id}: Unknown marking criterion ID."
            )

        criterion_total = 0.0

        for criterion_grade in grade.criterion_grades:
            criterion = expected_criteria.get(
                criterion_grade.criterion_id
            )

            if criterion is None:
                continue

            if criterion_grade.awarded_marks > criterion.max_marks + TOLERANCE:
                errors.append(
                    f"{grade.question_id}: Criterion "
                    f"{criterion_grade.criterion_id} exceeds maximum marks."
                )

            criterion_total += criterion_grade.awarded_marks

        if grade.criterion_grades:
            if abs(criterion_total - grade.awarded_marks) > TOLERANCE:
                errors.append(
                    f"{grade.question_id}: Criterion marks do not sum "
                    "to the question awarded marks."
                )

        if grade.is_fully_correct and abs(
            grade.awarded_marks - question.max_marks
        ) > TOLERANCE:
            errors.append(
                f"{grade.question_id}: Fully correct answer must receive "
                "full marks."
            )

    # Validate aggregate totals.
    if abs(result.total_awarded_marks - expected_total_awarded) > TOLERANCE:
        errors.append("Total awarded marks do not match question grades.")

    if abs(result.total_max_marks - expected_total_max) > TOLERANCE:
        errors.append("Total maximum marks do not match graded questions.")

    if result.total_awarded_marks > result.total_max_marks + TOLERANCE:
        errors.append("Total awarded marks exceed total maximum marks.")

    if expected_total_max > 0:
        expected_percentage = (
            expected_total_awarded / expected_total_max
        ) * 100

        if abs(result.percentage - expected_percentage) > 0.1:
            errors.append("Percentage does not match the grading totals.")

    return errors
