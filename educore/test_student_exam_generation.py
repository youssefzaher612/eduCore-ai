from __future__ import annotations

import json
import os

import httpx
import pytest

from educore.config import settings
from educore.rag.generator import OpenRouterGenerator
from educore.services.student_service import StudentLearningService
from educore.structured_output import parse_json_object


def validate(item, kind):
    return StudentLearningService._validate_generated_questions(
        [item], "Overfitting", "Medium", 1, {"Overfitting"}, expected_types=[kind]
    )[0]


def test_mcq_accepts_letter_answer_and_validates_four_distinct_options():
    q = validate({"question": "What is overfitting?", "options": ["A", "B", "C", "D"], "correct_answer": "C"}, "mcq")
    assert q["correct_index"] == 2
    assert q["question_type"] == "mcq"


def test_true_false_does_not_need_options():
    assert validate({"question": "Overfitting harms generalization.", "correct_answer": True}, "true_false")["correct_answer"] is True


def test_short_answer_does_not_need_options():
    q = validate({"question": "Define overfitting.", "expected_answer": "Poor generalization", "acceptable_alternatives": ["fails on unseen data"]}, "short_answer")
    assert q["acceptable_alternatives"] == ["fails on unseen data"]


def test_essay_does_not_need_options_and_keeps_rubric():
    q = validate({"question": "Explain overfitting.", "model_answer": "Learns training noise.", "grading_rubric": ["Definition", "Example"]}, "essay")
    assert q["grading_rubric"] == ["Definition", "Example"]


def test_invalid_item_does_not_discard_another_valid_question():
    questions = [
        {"question": "invalid MCQ", "options": ["A", "B"], "correct_index": 0},
        {"question": "Overfitting definition?", "options": ["A", "B", "C", "D"], "correct_index": 1},
    ]
    result = StudentLearningService._validate_generated_questions(questions, "Overfitting", "Easy", 2, {"Overfitting"})
    assert len(result) == 1
    assert result[0]["question"].startswith("Overfitting")


def test_markdown_json_and_top_level_single_question_normalization():
    payload = parse_json_object('```json\n{"question":"Q","options":["a","b","c","d"],"correct_index":0}\n```')
    result = StudentLearningService._validate_generated_questions([payload], "Overfitting", "Easy", 1, {"Overfitting"})
    assert len(result) == 1


def test_truncated_json_is_classified():
    with pytest.raises(ValueError, match="truncated JSON"):
        parse_json_object('{"questions":[{"question":"unfinished')


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_LOCAL_OLLAMA_TESTS") != "1", reason="set RUN_LOCAL_OLLAMA_TESTS=1 for real local inference")
async def test_real_ollama_qwen_student_question_types():
    async with httpx.AsyncClient(timeout=3) as client:
        response = await client.get(settings.OLLAMA_BASE_URL.rstrip("/") + "/api/tags")
        response.raise_for_status()
        installed = {item["name"] for item in response.json().get("models", [])}
    assert settings.OLLAMA_CHAT_MODEL in installed

    prompts = {
        "mcq": 'Return JSON exactly as {"title":"Overfitting","questions":[{"question_type":"mcq","question":"...","options":["a","b","c","d"],"correct_index":0,"explanation":"..."}]}. Topic overfitting.',
        "true_false": 'Return JSON exactly as {"title":"Overfitting","questions":[{"question_type":"true_false","question":"...","correct_answer":true,"explanation":"..."}]}. Topic overfitting.',
        "short_answer": 'Return JSON exactly as {"title":"Overfitting","questions":[{"question_type":"short_answer","question":"...","expected_answer":"...","explanation":"..."}]}. Topic overfitting.',
        "essay": 'Create one essay question. Keep model answer under 25 words, rubric exactly 2 criteria. Return JSON exactly as {"title":"Overfitting","questions":[{"question_type":"essay","question":"...","model_answer":"...","grading_rubric":["...","..."],"explanation":"..."}]}. Topic overfitting.',
    }
    generator = OpenRouterGenerator()
    for kind, prompt in prompts.items():
        raw = await generator.generate(
            [{"role": "user", "content": prompt}], max_tokens=900,
            temperature=0.2, response_format={"type": "json_object"}, num_ctx=4096,
        )
        payload = parse_json_object(raw)
        items = payload.get("questions") or ([payload] if payload.get("question") else [])
        validated = StudentLearningService._validate_generated_questions(
            items, "Overfitting", "Medium", 1, {"Overfitting"}, expected_types=[kind]
        )
        assert validated[0]["question_type"] == kind

    # Exercise the reported failure mode: consecutive one-question local
    # batches must not collapse to the model's first repeated response.
    previous: set[str] = set()
    focuses = ["definition and purpose", "how the main process works", "a key component or step"]
    for index, focus in enumerate(focuses):
        prompt = (
            f"Create one MCQ about overfitting, question {index + 1} of 3. "
            f"Focus on {focus}; make the angle different from these questions: {sorted(previous)}. "
            "Return JSON exactly as {\"questions\":[{\"question_type\":\"mcq\",\"question\":\"...\","
            "\"options\":[\"a\",\"b\",\"c\",\"d\"],\"correct_index\":0,\"explanation\":\"...\"}]}."
        )
        for attempt in range(4):
            raw = await generator.generate(
                [{"role": "user", "content": prompt}], max_tokens=520,
                temperature=0.45 if attempt == 0 else 0.7,
                response_format={"type": "json_object"}, num_ctx=4096,
            )
            payload = parse_json_object(raw)
            items = payload.get("questions") or ([payload] if payload.get("question") else [])
            try:
                question = StudentLearningService._validate_generated_questions(
                    items, "Overfitting", "Medium", 1, {"Overfitting"},
                    existing_questions=previous, expected_types=["mcq"],
                )[0]
                previous.add(" ".join(question["question"].casefold().split()))
                break
            except ValueError as exc:
                prompt += f" Retry with a different angle; validation issue: {exc}."
        else:
            pytest.fail(f"local Qwen could not complete question {index + 1} of 3")
    assert len(previous) == 3
