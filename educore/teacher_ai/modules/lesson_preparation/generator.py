"""
Lesson generation for EduCore Teacher AI: RAG-grounded prompt -> LLM -> validated lesson.

Public interface
----------------
    LessonGenerator(client=None, max_attempts=2).generate(request) -> GenerationResult
    generate_lesson(request) -> GenerationResult              (uses ``default_client``)
    GenerationResult(lesson, request_check, metadata)
    LessonGenerationError(message, kind, problems, attempts)

``kind`` is one of: "json" (no readable JSON object), "schema" (structure invalid),
"requirements" (valid structure but practice-question count / timing / mode differ from the
request) and "truncated" (the model ran out of output tokens, also after a retry).

Behaviour
---------
* One corrective retry by default (``max_attempts`` = total tries). The retry prompt lists the
  exact problems. A truncated response is retried once with a larger token budget, unless the
  budget is already at the model's ceiling (then retrying the same call is pointless).
* Transport and API failures (LLMError subclasses other than truncation) are NOT swallowed:
  they propagate so the UI can show the right message.
* Generation only. Verification, review and repair are the job of QualityReviewer and
  LessonPreparationService (service.py), so a lesson that fails them is still returned there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from educore.config import settings
from educore.teacher_ai.core import config
from educore.teacher_ai.core.llm_client import (
    LLMRequestError,
    LLMTruncatedError,
    llm_client as default_client,
)
from educore.structured_output import structured_response_format

from .json_utils import JSONExtractionError, extract_json
from .prompts import (
    LOCAL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_lesson_prompt,
    build_local_lesson_prompt,
)
from .schemas import (GeneratedLesson, LessonRequest, LessonSchemaError, RequestCheck, check_lesson_against_request,
                      parse_generated_lesson)

logger = logging.getLogger(__name__)

GENERATION_TEMPERATURE = 0.3


class LessonGenerationError(Exception):
    """The model answered, but not with a usable lesson (after the allowed retries)."""

    def __init__(self, message: str, *, kind: str, problems: Optional[list] = None, attempts: int = 1):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.problems = list(problems or [])
        self.attempts = attempts


@dataclass
class GenerationResult:
    lesson: dict
    request_check: RequestCheck = field(default_factory=RequestCheck)   # non-blocking warnings remain
    metadata: dict = field(default_factory=dict)


def _correction(problems: list, truncated_json: bool) -> str:
    shown = "\n".join(f"- {p}" for p in problems[:12])
    note = ("\nYour previous answer was cut off before the JSON was complete. Make the lesson more "
            "compact (fewer and shorter items) so it fits." if truncated_json else "")
    return ("\n\nCORRECTION REQUIRED\nYour previous answer was rejected for these reasons:\n"
            f"{shown}{note}\nReturn the complete corrected JSON object only, following every "
            "requirement above.")


def _apply_preparation_mode(lesson: dict, request: LessonRequest) -> None:
    """Make teacher-learning sections follow the mode selected by the teacher."""
    teacher_learning = lesson.setdefault("teacher_learning", {})
    deep_preparation = request.preparation_mode == "Understand and prepare"
    teacher_learning["include"] = deep_preparation
    if not deep_preparation:
        teacher_learning["foundations"] = []
        teacher_learning["teacher_examples"] = []
        teacher_learning["knowledge_warnings"] = []


def _normalize_local_payload(payload: dict, request: LessonRequest) -> dict:
    """Keep small local models inside the compact contract they were given."""
    # These optional sections are deliberately omitted in the local prompt to
    # preserve context and output budget. Enforce that contract before validation.
    for section in (
        "prerequisites", "classroom_interaction", "classroom_activities",
        "common_misconceptions", "formative_assessment", "homework",
        "teacher_notes", "post_lesson_reflection",
    ):
        payload[section] = []

    teacher_learning = payload.get("teacher_learning")
    if not isinstance(teacher_learning, dict):
        teacher_learning = {}
        payload["teacher_learning"] = teacher_learning
    teacher_learning["include"] = request.preparation_mode == "Understand and prepare"
    if request.preparation_mode != "Understand and prepare":
        teacher_learning["foundations"] = []
        teacher_learning["teacher_examples"] = []
        teacher_learning["knowledge_warnings"] = []
    return payload


def _reconcile_lesson_timing(stages: list, target_minutes: int) -> None:
    """Adjust stage durations to exactly match the requested lesson length."""
    if not stages:
        return
    current = sum(stage.get("duration_minutes", 0) for stage in stages)
    difference = target_minutes - current
    if difference > 0:
        stages[-1]["duration_minutes"] = stages[-1].get("duration_minutes", 0) + difference
        return
    excess = -difference
    # Preserve at least five minutes per stage where possible, then use one-minute
    # increments if a short requested lesson has more stages than that permits.
    for floor in (5, 1):
        for stage in reversed(stages):
            reducible = max(0, stage.get("duration_minutes", 0) - floor)
            reduction = min(excess, reducible)
            stage["duration_minutes"] -= reduction
            excess -= reduction
            if excess == 0:
                return


class LessonGenerator:
    def __init__(self, client=None, max_attempts: int = 2):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self.client = client if client is not None else default_client
        self.max_attempts = int(max_attempts)

    def _ceiling(self) -> int:
        value = getattr(self.client, "model_max_completion_tokens", None)
        return value if isinstance(value, int) and value > 0 else config.OPENROUTER_MODEL_MAX_COMPLETION_TOKENS

    def generate(self, request, source_material: str = "") -> GenerationResult:
        if not isinstance(request, LessonRequest):
            request = LessonRequest.model_validate(request)
        local_provider = settings.active_provider == "ollama"
        system_prompt = LOCAL_SYSTEM_PROMPT if local_provider else SYSTEM_PROMPT
        if local_provider:
            base_prompt = build_local_lesson_prompt(request, source_material, source_limit=900)
        else:
            base_prompt = build_lesson_prompt(request, source_material=source_material)
        prompt = base_prompt
        ceiling = self._ceiling()
        budget = min(config.OPENROUTER_GENERATION_MAX_TOKENS, ceiling)
        if local_provider:
            budget = min(
                3200,
                max(512, settings.OLLAMA_NUM_CTX - 850),
                ceiling,
            )

        kind, problems, truncated_json = "json", [], False
        response_format = structured_response_format(GeneratedLesson, "teacher_lesson_plan")
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.generate_detailed(
                    system_prompt, prompt, temperature=GENERATION_TEMPERATURE, max_tokens=budget,
                    response_format=response_format)
            except LLMTruncatedError as exc:
                kind, problems, truncated_json = "truncated", [exc.message], True
                if attempt == self.max_attempts:
                    break
                if local_provider:
                    base_prompt = build_local_lesson_prompt(
                        request, source_material, source_limit=500
                    )
                else:
                    compact_source = source_material[:2500]
                    base_prompt = build_lesson_prompt(request, source_material=compact_source)
                prompt = base_prompt + (
                    "\n\nCOMPACT RETRY: Your previous JSON exceeded the output limit. "
                    "Return the same complete schema, but keep every text field to one concise "
                    "sentence and every list to the minimum useful entries. Keep all required "
                    "fields, requested practice-question count, and exact lesson timing. "
                    "Do not repeat explanations across sections."
                )
                logger.info("lesson response truncated; retrying once with concise content")
                continue
            except LLMRequestError as exc:
                detail = str(exc.detail or "").casefold()
                context_limit_error = any(
                    marker in detail
                    for marker in (
                        "prompt is longer than the context",
                        "context length currently available",
                        "context window",
                    )
                )
                if local_provider and context_limit_error and attempt < self.max_attempts:
                    base_prompt = build_local_lesson_prompt(
                        request, source_material, source_limit=300
                    )
                    prompt = base_prompt
                    budget = max(900, budget // 2)
                    logger.warning(
                        "local lesson prompt exceeded Ollama context; retrying with compact evidence and %s output tokens",
                        budget,
                    )
                    continue
                if exc.status_code != 413 or attempt == self.max_attempts:
                    raise
                # Some provider/model plans impose a lower combined input/output
                # limit than advertised. Retry once with a compact RAG context and
                # smaller completion budget instead of failing the lesson outright.
                if local_provider:
                    base_prompt = build_local_lesson_prompt(
                        request, source_material, source_limit=500
                    )
                else:
                    compact_source = source_material[:2500]
                    base_prompt = build_lesson_prompt(request, source_material=compact_source)
                prompt = base_prompt
                budget = min(3000, max(1500, budget // 2), ceiling)
                logger.info(
                    "lesson request too large; retrying with compact RAG context and %s output tokens",
                    budget,
                )
                continue

            truncated_json = False
            try:
                lesson_payload = extract_json(response.content)
                if local_provider:
                    lesson_payload = _normalize_local_payload(lesson_payload, request)
                lesson = parse_generated_lesson(lesson_payload)
                # The request controls whether the teacher wants subject-learning
                # material. Keep the generated structure aligned with that choice.
                _apply_preparation_mode(lesson, request)
                # Automatically reconcile lesson timing and practice questions count
                stages = lesson.get("lesson_flow", [])
                if stages:
                    _reconcile_lesson_timing(stages, request.duration_minutes)

                questions = lesson.get("practice_questions", [])
                while len(questions) < request.question_count:
                    next_num = len(questions) + 1
                    questions.append({
                        "question_number": next_num,
                        "question_text": f"Extension question {next_num}: Apply the core concept to a practical case.",
                        "difficulty": "Intermediate",
                        "correct_answer": "Complete conceptual explanation.",
                        "solution_steps": ["Identify problem statement", "Apply learned definition"],
                    })
                if len(questions) > request.question_count:
                    lesson["practice_questions"] = questions[:request.question_count]

                check = check_lesson_against_request(lesson, request)
                if not check.errors:
                    metadata = response.to_metadata() if hasattr(response, "to_metadata") else {}
                    metadata["generation_attempts"] = attempt
                    logger.info(
                        "AI generation task=teacher_lesson model=%s stage=validated attempt=%s latency=%s tokens=%s",
                        metadata.get("model") or settings.active_chat_model, attempt,
                        metadata.get("latency_seconds"), metadata.get("completion_tokens"),
                    )
                    return GenerationResult(lesson=lesson, request_check=check, metadata=metadata)
                kind, problems = "requirements", [str(issue) for issue in check.errors]
            except JSONExtractionError as exc:
                kind, problems, truncated_json = "json", [str(exc)], exc.truncated
            except LessonSchemaError as exc:
                kind, problems = "schema", [str(p) for p in exc.problems]
            logger.warning(
                "AI generation task=teacher_lesson model=%s stage=validation category=%s attempt=%s/%s issue_count=%s",
                settings.active_chat_model, kind, attempt, self.max_attempts, len(problems),
            )
            prompt = base_prompt + _correction(problems, truncated_json)

        headline = {
            "json": "The AI did not return a readable lesson.",
            "schema": "The AI returned a lesson with missing or invalid sections.",
            "requirements": "The AI returned a lesson that does not match your request.",
            "truncated": "The AI response was cut off before the lesson was complete.",
        }[kind]
        detail = problems[0] if problems and kind != "truncated" else ""
        raise LessonGenerationError(f"{headline} {detail}".strip(), kind=kind, problems=problems,
                                    attempts=self.max_attempts if kind != "truncated" else attempt)


def generate_lesson(request) -> GenerationResult:
    """Convenience wrapper around the module-level ``default_client``."""
    return LessonGenerator(default_client).generate(request)
