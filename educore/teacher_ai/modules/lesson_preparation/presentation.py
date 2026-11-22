"""
Presentation helpers shared by the Streamlit UI and exports (no Streamlit import here, so
everything in this module is unit-testable).

    section_title(key)                    'practice_questions' -> 'Practice questions'
    section_to_markdown(key, value)       Markdown for one lesson section
    lesson_to_markdown(lesson, title)     whole lesson as one Markdown document (teacher view)
    group_issues_by_section(issues)       {'practice_questions': [Issue, ...], '': [lesson-wide]}
    friendly_error(exc)                   -> (title, message, hint) safe to show to a teacher

Nothing here prints API keys, stack traces or raw provider payloads: user-facing error
text is produced only from the exception's own redacted ``message`` or from fixed wording.
"""

from __future__ import annotations

from typing import Any

from educore.teacher_ai.core import config
from educore.teacher_ai.core.config import ConfigurationError
from educore.teacher_ai.core.llm_client import (LLMAuthenticationError, LLMConfigurationError, LLMConnectionError,
                             LLMError, LLMRateLimitError, LLMRequestError, LLMTimeoutError,
                             LLMTruncatedError)

from .schemas import REQUIRED_SECTIONS

try:  # pydantic is a hard dependency; imported lazily so this module stays light
    from pydantic import ValidationError as _ValidationError
except Exception:  # pragma: no cover
    _ValidationError = ()

SECTION_LABELS = {
    "lesson_title": "Lesson title",
    "lesson_overview": "Lesson overview",
    "learning_objectives": "Learning objectives",
    "prerequisites": "Prerequisites",
    "teacher_learning": "Teacher learning (understand the topic first)",
    "teaching_strategy": "Teaching strategy",
    "lesson_flow": "Lesson flow and timing",
    "concept_explanations": "Concept explanations and board work",
    "worked_examples": "Worked examples",
    "classroom_interaction": "Classroom interaction scenarios",
    "classroom_activities": "Classroom activities",
    "common_misconceptions": "Common misconceptions",
    "formative_assessment": "Formative assessment",
    "practice_questions": "Practice questions",
    "lesson_summary": "Lesson summary and closure",
    "homework": "Homework",
    "differentiation": "Differentiation",
    "teacher_notes": "Teacher notes",
    "post_lesson_reflection": "Post-lesson reflection",
}

_TITLE_KEYS = ("title", "stage", "concept", "activity_name", "misconception", "situation",
               "question", "question_or_task", "checkpoint", "knowledge", "objective",
               "problem_or_example", "step", "say")

STATUS_LABELS = {
    "ready": "Ready to review and teach",
    "needs_attention": "Needs your attention before teaching",
    "blocked": "Do not use as written: errors were found",
}


def humanize(key: str) -> str:
    return str(key).replace("_", " ").strip().capitalize()


def section_title(key: str) -> str:
    return SECTION_LABELS.get(key, humanize(key))


def _clean(text: Any) -> str:
    # '$' would start LaTeX in Markdown renderers; line breaks stay inside one list item.
    return " ".join(str(text).split()).replace("$", "\\$")


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _item_title(item: dict, index: int) -> str:
    for key in _TITLE_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            text = _clean(value)
            return f"{index}. {text if len(text) <= 90 else text[:90] + '…'}"
    return f"Item {index}"


def _render(value: Any, depth: int = 0) -> list:
    pad = "  " * depth
    lines: list = []
    if isinstance(value, dict):
        for key, inner in value.items():
            if _is_empty(inner):
                continue
            label = humanize(key)
            if isinstance(inner, (dict, list)):
                lines.append(f"{pad}- **{label}**")
                lines.extend(_render(inner, depth + 1))
            elif isinstance(inner, bool):
                lines.append(f"{pad}- **{label}:** {'yes' if inner else 'no'}")
            else:
                lines.append(f"{pad}- **{label}:** {_clean(inner)}")
    elif isinstance(value, list):
        for index, inner in enumerate(value, start=1):
            if isinstance(inner, dict):
                lines.append(f"{pad}- **{_item_title(inner, index)}**")
                lines.extend(_render(inner, depth + 1))
            elif isinstance(inner, list):
                lines.extend(_render(inner, depth))
            elif not _is_empty(inner):
                lines.append(f"{pad}- {_clean(inner)}")
    elif not _is_empty(value):
        lines.append(f"{pad}{_clean(value)}")
    return lines


def section_to_markdown(key: str, value: Any) -> str:
    if isinstance(value, str):
        return _clean(value)
    return "\n".join(_render(value)) or "_Nothing in this section._"


def lesson_to_markdown(lesson: dict, title: str = "") -> str:
    out = [f"# {_clean(title or lesson.get('lesson_title', 'Lesson plan'))}"]
    for key in REQUIRED_SECTIONS:
        if key == "lesson_title" or key not in lesson:
            continue
        value = lesson[key]
        if key == "teacher_learning" and not (isinstance(value, dict) and value.get("include")):
            continue
        body = value
        if key == "teacher_learning" and isinstance(value, dict):
            body = {k: v for k, v in value.items() if k != "include"}
        out.append(f"\n## {section_title(key)}\n\n{section_to_markdown(key, body)}")
    extra = [k for k in lesson if k not in REQUIRED_SECTIONS]
    for key in extra:
        out.append(f"\n## {humanize(key)}\n\n{section_to_markdown(key, lesson[key])}")
    return "\n".join(out).strip() + "\n"


def group_issues_by_section(issues) -> dict:
    grouped: dict = {}
    for issue in issues:
        section = str(issue.path[0]) if issue.path else ""
        grouped.setdefault(section, []).append(issue)
    return grouped


def friendly_error(exc: BaseException):
    """-> (title, message, hint). Never includes secrets, tracebacks or raw provider text."""
    from .generator import LessonGenerationError  # local import avoids a circular import

    if isinstance(exc, LLMConfigurationError):
        return ("Setup needed", exc.message,
                f"Add {config.settings.LLM_PROVIDER.upper()}_API_KEY to EduCore's .env file, then restart the app.")
    if isinstance(exc, LLMAuthenticationError):
        return ("The AI service rejected the API key", exc.message,
                f"Check {config.settings.LLM_PROVIDER.upper()}_API_KEY in your .env file.")
    if isinstance(exc, LLMRateLimitError):
        wait = exc.retry_after
        if wait and wait >= 60:
            minutes = int(wait // 60)
            seconds = int(wait % 60)
            return (
                "Groq's temporary TPM/RPM request limit has been reached",
                f"The provider asked EduCore to wait about {minutes}m {seconds}s before another request.",
                "The fast Groq model was tried as fallback. Wait for the quota reset before retrying.",
            )
        return (
            "Groq's temporary TPM/RPM request limit has been reached",
            exc.message,
            "Wait briefly for the quota window to reset, then try again.",
        )
    if isinstance(exc, LLMTruncatedError):
        return ("The lesson was too long to finish",
                "The AI ran out of space before it completed the lesson.",
                "Try a shorter duration or fewer practice questions, then generate the lesson again.")
    if isinstance(exc, (LLMTimeoutError, LLMConnectionError)):
        return ("Could not reach the AI service", exc.message, "Check your internet connection and try again.")
    if isinstance(exc, LLMRequestError):
        if exc.status_code == 413:
            return (
                "The lesson is still above the provider's request limit",
                "EduCore already retries with a shorter RAG excerpt and a smaller response budget.",
                "Choose one shorter topic, reduce the lesson duration or practice-question count, "
                "and restart the API so it loads the latest model settings.",
            )
        return ("The AI service could not accept the request", exc.message, "")
    if isinstance(exc, LLMError):
        return ("The AI service had a problem", exc.message, "Try again in a moment.")
    if isinstance(exc, LessonGenerationError):
        return ("The lesson could not be completed", exc.message,
                "Try again. If it keeps happening, simplify the topic or the extra instructions.")
    if isinstance(exc, ConfigurationError):
        return ("Configuration problem", str(exc), "Check the values in your .env file.")
    if _ValidationError and isinstance(exc, _ValidationError):
        return ("Please check the form", "Some lesson details are not valid.",
                "Review the subject, grade, topic and numbers, then try again.")
    return ("Something went wrong", "An unexpected error stopped the lesson preparation.",
            "Try again. If it repeats, check the application log.")
