
from .schemas import (
    AssessmentRequest,
    AssessmentResult,
    GeneratedAssessment,
    QuestionType,
)

from .math_validator import validate_linear_mcq

TOLERANCE = 0.001


def validate_assessment(
    request: AssessmentRequest,
    result: AssessmentResult,
) -> list[str]:
    """Validate assessment structure and request requirements."""

    errors = []
    questions = result.questions

    # 1. Question count
    if len(questions) != request.question_count:
        errors.append(
            f"Expected {request.question_count} questions, "
            f"got {len(questions)}."
        )

    # 2. Total marks
    if abs(result.total_marks - request.total_marks) > TOLERANCE:
        errors.append(
            f"Expected total marks {request.total_marks}, "
            f"got {result.total_marks}."
        )

    actual_marks = sum(q.marks for q in questions)

    if abs(actual_marks - request.total_marks) > TOLERANCE:
        errors.append(
            f"Question marks sum to {actual_marks}, "
            f"expected {request.total_marks}."
        )

    # 3. Question types
    allowed_types = set(request.question_types)

    for question in questions:
        if question.question_type not in allowed_types:
            errors.append(
                f"Question {question.question_number} uses "
                f"unrequested type: {question.question_type.value}."
            )

    # 4. Sequential numbering
    numbers = [q.question_number for q in questions]
    expected_numbers = list(range(1, len(questions) + 1))

    if numbers != expected_numbers:
        errors.append(
            "Question numbering must be sequential from 1."
        )

    # 5. Difficulty
    if request.difficulty.value != "mixed":
        for question in questions:
            if question.difficulty != request.difficulty:
                errors.append(
                    f"Question {question.question_number} "
                    "has unexpected difficulty."
                )

    # 6. Question validation
    for question in questions:
        if question.question_type == QuestionType.MCQ:
            if len(question.options) < 2:
                errors.append(
                    f"Question {question.question_number} "
                    "must have at least two options."
                )
                continue

            option_ids = [option.id for option in question.options]

            if len(option_ids) != len(set(option_ids)):
                errors.append(
                    f"Question {question.question_number} "
                    "has duplicate option IDs."
                )

            if (
                question.correct_answer is not None
                and question.correct_answer not in option_ids
            ):
                errors.append(
                    f"Question {question.question_number} "
                    "has a correct answer not present in its options."
                )

            option_texts = [
                option.text.strip().casefold()
                for option in question.options
            ]

            if len(option_texts) != len(set(option_texts)):
                errors.append(
                    f"Question {question.question_number} "
                    "has duplicate option texts."
                )

        elif question.options:
            errors.append(
                f"Question {question.question_number} "
                "has options but is not an MCQ."
            )

    # 7. Duration
    if result.duration_minutes != request.duration_minutes:
        errors.append(
            f"Expected duration {request.duration_minutes} minutes, "
            f"got {result.duration_minutes}."
        )

    return errors


def validate_generated_assessment(
    request: AssessmentRequest,
    result: GeneratedAssessment,
) -> list[str]:
    """Validate student and teacher versions and their consistency."""

    errors = []

    student = result.student_version
    teacher = result.teacher_version

    # 1. Validate each version independently
    errors.extend(
        f"Student version: {error}"
        for error in validate_assessment(request, student)
    )

    errors.extend(
        f"Teacher version: {error}"
        for error in validate_assessment(request, teacher)
    )

    # 2. Compare assessment-level fields
    if student.title != teacher.title:
        errors.append("Assessment titles differ between versions.")

    if student.instructions != teacher.instructions:
        errors.append("Assessment instructions differ between versions.")

    if abs(student.total_marks - teacher.total_marks) > TOLERANCE:
        errors.append("Assessment total marks differ between versions.")

    if student.duration_minutes != teacher.duration_minutes:
        errors.append("Assessment durations differ between versions.")

    # 3. Compare question counts
    if len(student.questions) != len(teacher.questions):
        errors.append(
            "Student and teacher versions have different question counts."
        )

    # 4. Compare corresponding questions
    for index, (student_q, teacher_q) in enumerate(
        zip(student.questions, teacher.questions),
        start=1,
    ):
        prefix = f"Question {index}"

        if student_q.question_number != teacher_q.question_number:
            errors.append(f"{prefix}: Question numbers differ.")

        if student_q.question_type != teacher_q.question_type:
            errors.append(f"{prefix}: Question types differ.")

        if student_q.difficulty != teacher_q.difficulty:
            errors.append(f"{prefix}: Difficulties differ.")

        if student_q.question_text != teacher_q.question_text:
            errors.append(f"{prefix}: Question texts differ.")

        if abs(student_q.marks - teacher_q.marks) > TOLERANCE:
            errors.append(f"{prefix}: Marks differ.")

        student_options = [
            (option.id, option.text)
            for option in student_q.options
        ]

        teacher_options = [
            (option.id, option.text)
            for option in teacher_q.options
        ]

        if student_options != teacher_options:
            errors.append(f"{prefix}: Options differ between versions.")

    # 5. Student version must not reveal answers
    for question in student.questions:
        if question.correct_answer is not None:
            errors.append(
                f"Student question {question.question_number} "
                "exposes the correct answer."
            )

        if question.explanation.strip():
            errors.append(
                f"Student question {question.question_number} "
                "exposes an explanation."
            )

        if question.marking_scheme:
            errors.append(
                f"Student question {question.question_number} "
                "exposes a marking scheme."
            )

    # 6. Teacher MCQs must have valid answers
    for question in teacher.questions:
        if question.question_type == QuestionType.MCQ:
            option_ids = [option.id for option in question.options]

            if not question.correct_answer:
                errors.append(
                    f"Teacher question {question.question_number} "
                    "is missing its correct answer."
                )
            elif question.correct_answer not in option_ids:
                errors.append(
                    f"Teacher question {question.question_number} "
                    "has an invalid correct answer."
                )
    # 7. Mathematical validation for supported linear-equation MCQs
    for question in teacher.questions:
        errors.extend(validate_linear_mcq(question))

    return errors
