from typing import Any
from pydantic import ValidationError

from educore.structured_output import StructuredOutputError, parse_json_object
from educore.teacher_ai.core.llm_client import LLMTruncatedError, llm_client

from .schemas import GradingRequest, GradingResult
from .prompts import (
    GRADING_SYSTEM_PROMPT,
    build_grading_prompt,
)
from .validator import validate_grading_result


def _parse_json_response(response: Any) -> dict:
    """Parse a JSON object from the LLM response."""
    try:
        return parse_json_object(response)
    except StructuredOutputError as exc:
        raise ValueError("AI did not return a complete grading JSON object.") from exc


def _reconcile_grading_data(request: GradingRequest, data: dict) -> dict:
    """Normalize LLM grading arithmetic and criteria to prevent validation rejections."""
    questions_by_id = {q.question_id: q for q in request.questions}
    answers_by_id = {answer.question_id: answer.answer for answer in request.student_answers}
    total_awarded = 0.0
    total_max = 0.0

    question_grades = data.get("question_grades", [])
    for qg in question_grades:
        qid = qg.get("question_id")
        q = questions_by_id.get(qid)
        if not q:
            continue
        qg["max_marks"] = q.max_marks
        awarded = float(qg.get("awarded_marks", 0.0))
        awarded = max(0.0, min(awarded, q.max_marks))

        expected = " ".join((q.expected_answer or "").split()).casefold()
        submitted = " ".join(answers_by_id.get(qid, "").split()).casefold()
        exact_match = bool(expected) and submitted == expected
        if exact_match:
            awarded = q.max_marks
            qg["is_fully_correct"] = True
            qg["feedback"] = "Your answer matches the expected answer."
            qg["criterion_grades"] = [
                {
                    "criterion_id": criterion.criterion_id,
                    "awarded_marks": criterion.max_marks,
                    "feedback": "Matches the expected answer.",
                }
                for criterion in q.marking_criteria
            ]

        expected_criteria = {c.criterion_id: c for c in q.marking_criteria}
        if exact_match:
            pass
        elif not expected_criteria:
            qg["criterion_grades"] = []
        else:
            filtered_crits = []
            crit_sum = 0.0
            for cg in qg.get("criterion_grades", []):
                cid = cg.get("criterion_id")
                if cid in expected_criteria:
                    c_awarded = min(float(cg.get("awarded_marks", 0.0)), expected_criteria[cid].max_marks)
                    cg["awarded_marks"] = c_awarded
                    crit_sum += c_awarded
                    filtered_crits.append(cg)
            qg["criterion_grades"] = filtered_crits
            if filtered_crits:
                awarded = crit_sum

        if qg.get("is_fully_correct"):
            if abs(awarded - q.max_marks) > 0.01:
                if awarded >= q.max_marks * 0.9:
                    awarded = q.max_marks
                else:
                    qg["is_fully_correct"] = False
        elif awarded >= q.max_marks - 0.01:
            qg["is_fully_correct"] = True

        qg["awarded_marks"] = round(awarded, 2)
        total_awarded += qg["awarded_marks"]
        total_max += q.max_marks

    data["total_awarded_marks"] = round(total_awarded, 2)
    data["total_max_marks"] = round(total_max, 2)
    data["percentage"] = round((total_awarded / total_max * 100) if total_max > 0 else 0.0, 1)
    return data


def grade_submission(
    request: GradingRequest,
    max_tokens: int = 4096,
) -> GradingResult:
    """Grade and validate a student submission."""

    if not isinstance(request, GradingRequest):
        request = GradingRequest.model_validate(request)

    request_data = request.model_dump(mode="json")
    user_prompt = build_grading_prompt(request_data)
    for attempt in range(2):
        try:
            response = llm_client.generate(
                system_prompt=GRADING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except LLMTruncatedError as exc:
            if attempt:
                raise ValueError("The grading response was incomplete after one retry.") from exc
            user_prompt += "\n\nRETRY: Return one complete JSON object. Keep feedback concise and include every submitted question ID."
            continue

        try:
            parsed_response = _parse_json_response(response)
            parsed_response = _reconcile_grading_data(request, parsed_response)
            result = GradingResult.model_validate(parsed_response)
            errors = validate_grading_result(request, result)
            if errors:
                raise ValueError("; ".join(errors))
            return result
        except (ValueError, TypeError, ValidationError) as exc:
            if attempt:
                raise ValueError("AI grading output was invalid after one correction.") from exc
            user_prompt += (
                "\n\nCORRECTION: Return valid JSON with a question_grades entry for every "
                "submitted question ID, valid criterion IDs, bounded marks, and consistent totals."
            )

    raise ValueError("AI grading output could not be validated.")
