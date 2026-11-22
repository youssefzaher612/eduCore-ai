"""RAG-grounded assessment generation, split to stay within provider token limits."""

import logging

from pydantic import ValidationError

from educore.config import settings
from educore.structured_output import parse_json_object, structured_response_format
from educore.teacher_ai.core.llm_client import LLMRequestError, LLMTruncatedError, llm_client
from .prompts import SYSTEM_PROMPT, build_assessment_prompt
from .schemas import (
    AssessmentQuestion, AssessmentRequest, AssessmentResult, Difficulty,
    GeneratedAssessment, GeneratedEssay, GeneratedMCQ, GeneratedProblemSolving,
    GeneratedShortAnswer, GeneratedTrueFalse, MarkingPoint, QuestionType,
    assessment_output_schema, generated_question_adapter,
)
from .validator import validate_generated_assessment

logger = logging.getLogger(__name__)


class AssessmentGenerationError(Exception):
    """Raised when assessment generation or validation fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _normalize_question_type(value):
    """Normalize common spelling variants without guessing a missing mixed type."""
    if isinstance(value, QuestionType):
        return value.value
    if not isinstance(value, str):
        return value
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "multiple_choice": "mcq", "multiplechoice": "mcq", "true/false": "true_false",
        "truefalse": "true_false", "shortanswer": "short_answer", "essay_question": "essay",
        "problem_solving_question": "problem_solving",
    }
    return aliases.get(normalized, normalized)


def _prepare_raw_question(raw_question: dict, request: AssessmentRequest) -> dict:
    if not isinstance(raw_question, dict):
        raise ValueError("Each question must be a JSON object.")
    question = dict(raw_question)
    question["question_type"] = _normalize_question_type(question.get("question_type"))
    if question.get("topic") and question["topic"].strip().casefold() != request.topic.strip().casefold():
        raise ValueError("Question topic does not match the requested course topic.")

    if question["question_type"] == QuestionType.MCQ.value:
        options = question.get("options")
        if isinstance(options, list):
            question["options"] = [
                item if isinstance(item, dict) else {"id": chr(65 + i), "text": str(item)}
                for i, item in enumerate(options)
            ]
            answer = question.get("correct_answer")
            if isinstance(answer, int) and not isinstance(answer, bool) and 0 <= answer < len(question["options"]):
                question["correct_answer"] = str(question["options"][answer].get("id", ""))
    if question["question_type"] == QuestionType.ESSAY.value:
        rubric = question.get("grading_rubric")
        if isinstance(rubric, list):
            question["grading_rubric"] = [
                item if isinstance(item, dict) else {"description": str(item)}
                for item in rubric
            ]
    return question


def _split_marks(total: float, count: int) -> list[float]:
    """Allocate marks in Python and put rounding residue on the final item."""
    if count < 1:
        return []
    each = round(total / count, 4)
    marks = [each] * (count - 1)
    marks.append(round(total - sum(marks), 4))
    if any(mark <= 0 for mark in marks):
        raise ValueError("The requested total marks are too small for this question count.")
    return marks


def _rubric_points(criteria, maximum: float) -> list[MarkingPoint]:
    if not criteria:
        raise ValueError("Essay questions must include measurable grading rubric criteria.")
    points = _split_marks(maximum, len(criteria))
    return [MarkingPoint(description=item.description, marks=point) for item, point in zip(criteria, points)]


def _typed_question_to_legacy(question, number: int, marks: float, request: AssessmentRequest) -> AssessmentQuestion:
    """Adapt a type-validated question to the existing teacher/student UI contract."""
    common = {
        "question_number": number,
        "question_id": f"question_{number}",
        "topic": request.topic,
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "question_text": question.question_text,
        "marks": marks,
        "maximum_score": marks,
    }
    if isinstance(question, GeneratedMCQ):
        options = question.options
        answer = question.correct_answer.strip()
        matching = next((option for option in options if option.id.casefold() == answer.casefold()), None)
        if matching is None:
            matching = next((option for option in options if option.text.casefold() == answer.casefold()), None)
        if matching is None:
            raise ValueError("MCQ correct_answer must match an option ID or option text.")
        return AssessmentQuestion(**common, options=options, correct_answer=matching.id,
                                  explanation=question.explanation,
                                  marking_scheme=[MarkingPoint(description="Correct answer", marks=marks)])
    if isinstance(question, GeneratedTrueFalse):
        answer_text = "True" if question.correct_answer else "False"
        return AssessmentQuestion(**common, correct_answer=question.correct_answer,
                                  explanation=question.explanation,
                                  marking_scheme=[MarkingPoint(description=f"Correct answer: {answer_text}", marks=marks)])
    if isinstance(question, GeneratedEssay):
        rubric = _rubric_points(question.grading_rubric, marks)
        return AssessmentQuestion(**common, correct_answer=question.model_answer, model_answer=question.model_answer,
                                  grading_rubric=rubric, marking_scheme=rubric)
    if isinstance(question, GeneratedShortAnswer):
        return AssessmentQuestion(**common, correct_answer=question.expected_answer,
                                  expected_answer=question.expected_answer,
                                  acceptable_alternatives=question.acceptable_alternatives,
                                  explanation=question.explanation,
                                  marking_scheme=[MarkingPoint(description="Expected answer", marks=marks)])
    if isinstance(question, GeneratedProblemSolving):
        points = _split_marks(marks, len(question.solution_steps))
        marking = [MarkingPoint(description=step, marks=value) for step, value in zip(question.solution_steps, points)]
        return AssessmentQuestion(**common, correct_answer=question.expected_answer,
                                  expected_answer=question.expected_answer, explanation=question.explanation,
                                  marking_scheme=marking)
    raise ValueError(f"Unsupported question payload: {type(question).__name__}.")


def _parse_teacher_assessment(request: AssessmentRequest, data: dict) -> AssessmentResult:
    """Validate each question against its own schema before frontend adaptation."""
    if "teacher_version" in data:
        data = data["teacher_version"]
    if not isinstance(data, dict):
        raise ValueError("The model response must contain a teacher assessment object.")
    if data.get("duration_minutes") != request.duration_minutes:
        raise ValueError("Assessment duration does not match the requested duration.")
    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list) or len(raw_questions) != request.question_count:
        actual = len(raw_questions) if isinstance(raw_questions, list) else 0
        raise ValueError(f"Expected exactly {request.question_count} questions, got {actual}.")

    allowed = {item.value for item in request.question_types}
    parsed = []
    for index, item in enumerate(raw_questions, start=1):
        prepared = _prepare_raw_question(item, request)
        question = generated_question_adapter.validate_python(prepared)
        qtype = question.question_type.value
        if qtype not in allowed:
            raise ValueError(f"Question {index} uses unrequested type {qtype}.")
        if request.difficulty != Difficulty.MIXED:
            # The teacher's fixed difficulty is authoritative metadata. The item
            # content was requested at that level; normalize an inconsistent label.
            question.difficulty = request.difficulty
        parsed.append(question)

    marks = _split_marks(request.total_marks, len(parsed))
    questions = [
        _typed_question_to_legacy(question, index, marks[index - 1], request)
        for index, question in enumerate(parsed, start=1)
    ]
    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        title = f"{request.topic} {request.assessment_type.value.title()}"
    instructions = data.get("instructions", [])
    if not isinstance(instructions, list) or not all(isinstance(value, str) for value in instructions):
        raise ValueError("Assessment instructions must be a list of strings.")
    return AssessmentResult(title=title.strip(), instructions=instructions, questions=questions,
                            total_marks=request.total_marks, duration_minutes=request.duration_minutes)

def _one_batch(
    request: AssessmentRequest,
    source_material: str,
    max_tokens: int,
    *,
    max_correction_attempts: int = 1,
) -> AssessmentResult:
    """Generate a typed teacher batch and repair only from the validation failure."""
    payload = request.model_dump(mode="json")
    source = (source_material or "")[:3000]
    prompt = build_assessment_prompt(payload, source_material=source)
    local = settings.active_provider == "ollama"
    local_ceiling = max(512, settings.OLLAMA_NUM_CTX - 1400) if local else 4096
    budget = min(max_tokens, 4096, local_ceiling, max(1200, 700 + request.question_count * 450))
    last_error = None
    schema_model = assessment_output_schema(tuple(item.value for item in request.question_types))
    response_format = structured_response_format(schema_model, "teacher_assessment")

    total_attempts = max(1, int(max_correction_attempts) + 1)
    for attempt in range(total_attempts):
        attempt_number = attempt + 1
        try:
            response = llm_client.generate_detailed(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.0 if local else 0.2,
                max_tokens=budget,
                response_format=response_format,
            )
            raw_response = response.content
            logger.info(
                "AI generation task=teacher_exam model=%s stage=response attempt=%s/%s latency=%.1fs tokens=%s",
                response.model or settings.active_chat_model, attempt_number, total_attempts,
                response.latency_seconds, response.completion_tokens,
            )
        except LLMTruncatedError as exc:
            last_error = exc
            logger.warning("AI generation task=teacher_exam model=%s stage=truncated attempt=%s/%s",
                           settings.active_chat_model, attempt_number, total_attempts)
            if attempt + 1 >= total_attempts:
                raise
            source = source[:1000]
            prompt = build_assessment_prompt(payload, source_material=source) + (
                "\n\nREPAIR: The previous response reached the output limit. Preserve the original task and "
                "all required type-specific fields, but shorten wording and return the complete object."
            )
            budget = max(512, budget - 300)
            continue
        except LLMRequestError as exc:
            detail = str(exc.detail or "").casefold()
            context_error = any(token in detail for token in ("context length", "context window", "longer than the context"))
            if not (exc.status_code == 413 or context_error) or attempt + 1 >= total_attempts:
                raise
            source = source[:1000]
            budget = max(512, budget // 2)
            prompt = build_assessment_prompt(payload, source_material=source) + (
                "\n\nREPAIR: The original prompt exceeded the local context limit. Keep the exact question count "
                "and type-specific fields; use only this shorter course excerpt."
            )
            logger.warning("AI generation task=teacher_exam model=%s stage=context_retry attempt=%s/%s",
                           settings.active_chat_model, attempt_number, total_attempts)
            continue

        try:
            data = parse_json_object(raw_response)
            data = schema_model.model_validate(data).model_dump(mode="python")
            teacher_version = _parse_teacher_assessment(request, data)

            student_data = teacher_version.model_dump(mode="json")
            for question in student_data["questions"]:
                question["correct_answer"] = None
                question["explanation"] = ""
                question["marking_scheme"] = []
                question["model_answer"] = None
                question["grading_rubric"] = []
                question["expected_answer"] = None
                question["acceptable_alternatives"] = []
            result = GeneratedAssessment(
                student_version=AssessmentResult.model_validate(student_data),
                teacher_version=teacher_version,
            )
            errors = validate_generated_assessment(request, result)
            if errors:
                raise AssessmentGenerationError("; ".join(errors[:8]))
            return teacher_version
        except (ValueError, TypeError, ValidationError, AssessmentGenerationError) as exc:
            last_error = exc
            error_text = str(exc)[:1000]
            logger.warning(
                "AI generation task=teacher_exam model=%s stage=validation category=%s attempt=%s/%s",
                settings.active_chat_model, type(exc).__name__, attempt_number, total_attempts,
            )
            if attempt + 1 >= total_attempts:
                break
            source = source[:1500]
            prompt = build_assessment_prompt(payload, source_material=source)
            prompt += (
                "\n\nREPAIR ONLY THE INVALID OUTPUT. Keep the original task, source, and question text where valid. "
                "Apply these exact validation errors, then return the complete corrected assessment object:\n- "
                + error_text
            )

    raise AssessmentGenerationError(
        f"Unable to generate a valid assessment after {total_attempts} attempts. "
        f"Last validation error: {str(last_error)[:500]}"
    ) from last_error


def _split_request(request: AssessmentRequest) -> tuple[AssessmentRequest, AssessmentRequest]:
    left_count = request.question_count // 2
    right_count = request.question_count - left_count
    left_marks = round(request.total_marks * left_count / request.question_count, 3)
    right_marks = request.total_marks - left_marks
    left = request.model_copy(update={"question_count": left_count, "total_marks": left_marks})
    right = request.model_copy(update={"question_count": right_count, "total_marks": right_marks})
    return left, right


def _generate_portion(
    request: AssessmentRequest,
    source_material: str,
    max_tokens: int,
    max_correction_attempts: int = 1,
) -> AssessmentResult:
    """Generate a portion; if it still hits the cap, divide it into smaller calls."""
    try:
        return _one_batch(request, source_material, max_tokens,
                          max_correction_attempts=max_correction_attempts)
    except LLMTruncatedError:
        if request.question_count <= 1:
            raise
    except LLMRequestError as exc:
        if exc.status_code != 413 or request.question_count <= 1:
            raise
    except AssessmentGenerationError:
        if request.question_count <= 1:
            raise

    left_request, right_request = _split_request(request)
    left = _generate_portion(left_request, source_material, max_tokens, max_correction_attempts)
    right = _generate_portion(right_request, source_material, max_tokens, max_correction_attempts)
    questions = [*left.questions, *right.questions]
    for number, question in enumerate(questions, start=1):
        question.question_number = number
        question.question_id = f"question_{number}"
    return AssessmentResult(
        title=left.title,
        instructions=left.instructions,
        questions=questions,
        total_marks=request.total_marks,
        duration_minutes=request.duration_minutes,
    )


def _generate_in_batches(
    request: AssessmentRequest,
    source_material: str,
    max_tokens: int,
    max_correction_attempts: int = 1,
) -> AssessmentResult:
    """Make predictable small calls so ordinary exams never hit the output cap."""
    batch_size = 4
    if request.question_count <= batch_size:
        return _generate_portion(request, source_material, max_tokens, max_correction_attempts)

    portions = []
    remaining_marks = request.total_marks
    remaining_questions = request.question_count
    while remaining_questions:
        count = min(batch_size, remaining_questions)
        marks = (
            remaining_marks
            if count == remaining_questions
            else round(request.total_marks * count / request.question_count, 3)
        )
        portion_request = request.model_copy(update={
            "question_count": count,
            "total_marks": marks,
        })
        portions.append(_generate_portion(portion_request, source_material, max_tokens,
                                          max_correction_attempts))
        remaining_marks -= marks
        remaining_questions -= count

    questions = [question for portion in portions for question in portion.questions]
    for number, question in enumerate(questions, start=1):
        question.question_number = number
        question.question_id = f"question_{number}"
    return AssessmentResult(
        title=portions[0].title,
        instructions=portions[0].instructions,
        questions=questions,
        total_marks=request.total_marks,
        duration_minutes=request.duration_minutes,
    )


def generate_assessment(
    request: AssessmentRequest,
    max_tokens: int = 4096,
    source_material: str = "",
    max_correction_attempts: int | None = None,
) -> GeneratedAssessment:
    """Generate in small calls, then build the student copy without answers."""
    if not isinstance(request, AssessmentRequest):
        request = AssessmentRequest.model_validate(request)

    if max_correction_attempts is None:
        max_correction_attempts = settings.ASSESSMENT_GENERATION_MAX_CORRECTION_ATTEMPTS
    if max_correction_attempts < 0:
        raise ValueError("max_correction_attempts cannot be negative.")
    teacher_version = _generate_in_batches(
        request, source_material, max_tokens, max_correction_attempts
    )
    student_data = teacher_version.model_dump(mode="json")
    for question in student_data["questions"]:
        question["correct_answer"] = None
        question["explanation"] = ""
        question["marking_scheme"] = []
        question["model_answer"] = None
        question["grading_rubric"] = []
        question["expected_answer"] = None
        question["acceptable_alternatives"] = []
    result = GeneratedAssessment(
        student_version=AssessmentResult.model_validate(student_data),
        teacher_version=teacher_version,
    )
    errors = validate_generated_assessment(request, result)
    if errors:
        raise AssessmentGenerationError("; ".join(errors[:8]))
    return result
