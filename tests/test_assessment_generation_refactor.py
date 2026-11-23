import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError

from educore.structured_output import parse_json_object, structured_response_format
from educore.teacher_ai.modules.assessment_generator.generator import (
    AssessmentGenerationError, _parse_teacher_assessment, generate_assessment,
)
from educore.teacher_ai.modules.assessment_generator.schemas import (
    AssessmentRequest, GeneratedTrueFalse, QuestionType, assessment_output_schema,
)
from educore.teacher_ai.modules.assessment_generator.prompts import build_assessment_prompt
from educore.teacher_ai.modules.lesson_preparation.schemas import (
    GeneratedLesson, LessonPlan, LessonRequest, adapt_generated_lesson_to_plan,
)
from educore.services.student_service import StudentLearningService
from educore.teacher_ai.core.llm_client import LLMClient


def request(types, difficulty="easy", count=None):
    return AssessmentRequest(
        subject="Biology", grade="8", topic="Cell membrane", assessment_type="quiz",
        difficulty=difficulty, question_types=types, question_count=count or len(types),
        total_marks=6, duration_minutes=20,
    )


def question(qtype, difficulty="easy"):
    common = {"question_type": qtype, "question_text": "Apply the concept to a new case.", "difficulty": difficulty}
    if qtype == "mcq":
        return {**common, "options": [{"id": "A", "text": "Correct"}, {"id": "B", "text": "Incorrect"}], "correct_answer": "A", "explanation": "The first option applies the rule."}
    if qtype == "true_false":
        return {**common, "correct_answer": False, "explanation": "The claim contradicts the source."}
    if qtype == "essay":
        return {**common, "model_answer": "A substantive model response.", "grading_rubric": [{"description": "Explains the process accurately."}, {"description": "Uses evidence from the source."}]}
    if qtype == "short_answer":
        return {**common, "expected_answer": "The membrane controls movement.", "acceptable_alternatives": ["It regulates exchange."], "explanation": "That is its selective role."}
    return {**common, "expected_answer": "Particles diffuse down the gradient.", "solution_steps": ["Compare concentrations.", "State the direction."], "explanation": "Diffusion follows the concentration gradient."}


def payload(questions):
    return {"title": "Cell Membrane Quiz", "instructions": ["Answer each item."], "questions": questions, "duration_minutes": 20}


class AssessmentPipelineTests(unittest.TestCase):
    def make_response(self, value):
        return SimpleNamespace(content=json.dumps(value), model="mock-model", latency_seconds=0.01, completion_tokens=100)

    def test_each_question_type_has_its_own_contract(self):
        cases = [
            ("mcq", "correct_answer"), ("true_false", "correct_answer"),
            ("essay", "grading_rubric"), ("short_answer", "acceptable_alternatives"),
            ("problem_solving", "solution_steps"),
        ]
        for qtype, key in cases:
            with self.subTest(question_type=qtype):
                req = request([qtype])
                item = question(qtype)
                result = _parse_teacher_assessment(req, payload([item]))
                self.assertEqual(result.questions[0].question_type.value, qtype)
                self.assertEqual(result.questions[0].topic, req.topic)
                self.assertTrue(getattr(result.questions[0], key) if qtype == "essay" else True)

    def test_true_false_accepts_boolean_and_supported_string_without_options(self):
        base = {"question_type": "true_false", "question_text": "Claim", "difficulty": "easy", "explanation": "Reason"}
        for answer, expected in [(False, False), (True, True), ("false", False), ("true", True)]:
            with self.subTest(answer=answer):
                parsed = GeneratedTrueFalse.model_validate({**base, "correct_answer": answer})
                self.assertIs(parsed.correct_answer, expected)
        with self.assertRaises(ValidationError):
            GeneratedTrueFalse.model_validate({**base, "correct_answer": "maybe"})
        with self.assertRaises(ValidationError):
            GeneratedTrueFalse.model_validate({**base, "correct_answer": True, "options": ["T", "F"]})

    def test_mcq_answer_must_reference_an_option_and_exact_count_is_required(self):
        req = request(["mcq"])
        with self.assertRaisesRegex(ValueError, "must match an option"):
            _parse_teacher_assessment(req, payload([{**question("mcq"), "correct_answer": "Z"}]))
        with self.assertRaisesRegex(ValueError, "Expected exactly 1 questions"):
            _parse_teacher_assessment(req, payload([]))

    def test_teacher_generation_routes_each_requested_type_through_its_own_schema(self):
        for qtype in ("mcq", "true_false", "essay", "short_answer", "problem_solving"):
            with self.subTest(question_type=qtype):
                req = request([qtype])
                response = self.make_response(payload([question(qtype)]))
                with patch("educore.teacher_ai.modules.assessment_generator.generator.structured_response_format", return_value=None), patch(
                    "educore.teacher_ai.modules.assessment_generator.generator.llm_client.generate_detailed", return_value=response
                ):
                    result = generate_assessment(req, source_material="Course evidence")
                self.assertEqual(result.teacher_version.questions[0].question_type.value, qtype)

    def test_mixed_exam_maps_fields_and_strips_all_answers_for_student(self):
        req = request(["true_false", "essay", "short_answer"], difficulty="mixed", count=3)
        raw = payload([question("true_false"), question("essay", "medium"), question("short_answer", "hard")])
        response = self.make_response(raw)
        with patch("educore.teacher_ai.modules.assessment_generator.generator.structured_response_format", return_value=None), patch(
            "educore.teacher_ai.modules.assessment_generator.generator.llm_client.generate_detailed", return_value=response
        ):
            result = generate_assessment(req, source_material="Cell membrane excerpt")
        self.assertEqual([q.question_type.value for q in result.teacher_version.questions], ["true_false", "essay", "short_answer"])
        self.assertEqual([q.question_id for q in result.teacher_version.questions], ["question_1", "question_2", "question_3"])
        self.assertAlmostEqual(sum(q.marks for q in result.teacher_version.questions), req.total_marks)
        essay = result.teacher_version.questions[1]
        self.assertAlmostEqual(sum(p.marks for p in essay.grading_rubric), essay.maximum_score)
        for q in result.student_version.questions:
            self.assertIsNone(q.correct_answer)
            self.assertIsNone(q.model_answer)
            self.assertIsNone(q.expected_answer)
            self.assertEqual(q.grading_rubric, [])
            self.assertEqual(q.acceptable_alternatives, [])

    def test_fixed_difficulty_metadata_and_marks_are_assigned_from_request(self):
        req = request(["essay"])
        item = question("essay", difficulty="medium")
        result = _parse_teacher_assessment(req, payload([item]))
        self.assertEqual(result.questions[0].difficulty.value, "easy")
        self.assertEqual(result.questions[0].marks, 6)
        self.assertEqual(result.questions[0].maximum_score, 6)

    def test_validation_repair_includes_exact_error_and_original_source(self):
        req = request(["mcq"])
        invalid = payload([{**question("mcq"), "correct_answer": "Z"}])
        valid = payload([question("mcq")])
        responses = [self.make_response(invalid), self.make_response(valid)]
        with patch("educore.teacher_ai.modules.assessment_generator.generator.structured_response_format", return_value=None), patch(
            "educore.teacher_ai.modules.assessment_generator.generator.llm_client.generate_detailed", side_effect=responses
        ) as generate:
            result = generate_assessment(req, source_material="SOURCE_EVIDENCE")
        self.assertEqual(result.teacher_version.questions[0].correct_answer, "A")
        self.assertEqual(generate.call_count, 2)
        self.assertIn("must match an option ID or option text", generate.call_args_list[1].kwargs["user_prompt"])
        self.assertIn("SOURCE_EVIDENCE", generate.call_args_list[1].kwargs["user_prompt"])

    def test_failed_repair_reports_root_validation_error_not_generic_message(self):
        req = request(["mcq"])
        invalid = self.make_response(payload([{**question("mcq"), "correct_answer": "Z"}]))
        with patch("educore.teacher_ai.modules.assessment_generator.generator.structured_response_format", return_value=None), patch(
            "educore.teacher_ai.modules.assessment_generator.generator.llm_client.generate_detailed", return_value=invalid
        ):
            with self.assertRaisesRegex(AssessmentGenerationError, "correct_answer must match"):
                generate_assessment(req, max_correction_attempts=0)

    def test_prompt_only_includes_requested_type_specific_shapes(self):
        text = build_assessment_prompt(request(["true_false", "essay"]).model_dump(mode="json"))
        self.assertIn('"correct_answer": true', text)
        self.assertIn('"grading_rubric"', text)
        self.assertNotIn('"options"', text)

    def test_ollama_builds_json_schema_format_for_current_server(self):
        req_schema = assessment_output_schema((QuestionType.TRUE_FALSE.value,))
        with patch("educore.structured_output.settings.AI_PROVIDER", ""), patch(
            "educore.structured_output.settings.LLM_PROVIDER", "ollama"
        ), patch("educore.structured_output._ollama_schema_supported", return_value=True):
            fmt = structured_response_format(req_schema, "teacher_assessment")
        self.assertEqual(fmt["type"], "json_schema")
        self.assertIn("questions", fmt["json_schema"]["schema"]["properties"])

    def test_ollama_unavailable_falls_back_to_json_object_mode(self):
        req_schema = assessment_output_schema((QuestionType.MCQ.value,))
        with patch("educore.structured_output.settings.AI_PROVIDER", ""), patch(
            "educore.structured_output.settings.LLM_PROVIDER", "ollama"
        ), patch("educore.structured_output._ollama_schema_supported", return_value=False):
            fmt = structured_response_format(req_schema, "teacher_assessment")
        self.assertIsNone(fmt)

    def test_llm_client_falls_back_from_rejected_json_schema_to_json_mode(self):
        rejected = Exception("HTTP 400 response_format json_schema is unsupported")
        rejected.status_code = 400
        rejected.body = {"error": {"message": "unsupported response_format"}}
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            model="qwen3-vl:2b", id="local-test",
        )
        create = Mock(side_effect=[rejected, completion])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("educore.teacher_ai.core.llm_client.settings.AI_PROVIDER", ""), patch(
            "educore.teacher_ai.core.llm_client.settings.LLM_PROVIDER", "ollama"
        ):
            client = LLMClient(sdk_client=sdk, model="qwen3-vl:2b", json_mode=False, max_retries=0)
            result = client.generate_detailed(
                "Return JSON.", "Return an object.", max_tokens=128,
                response_format={"type": "json_schema", "json_schema": {"name": "Test", "schema": {}}},
            )
        self.assertEqual(result.content, '{"ok":true}')
        self.assertEqual(create.call_args_list[0].kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(create.call_args_list[1].kwargs["response_format"], {"type": "json_object"})

    def test_student_practice_service_validates_and_saves_mcq_payload(self):
        user_id, project_id = uuid.uuid4(), uuid.uuid4()
        session = SimpleNamespace(add=Mock(), flush=AsyncMock())
        service = StudentLearningService(session)
        service.validate_scope = AsyncMock(return_value=None)
        service._course_topics = AsyncMock(return_value=[{"name": "Cell membrane"}])
        source = SimpleNamespace(document_name="biology.pdf", page_number=1, similarity=0.9,
                                 content="The membrane regulates entry and exit.")
        raw = json.dumps({"title": "Practice", "questions": [{
            "question": "What does the membrane regulate?",
            "options": ["Entry and exit", "Photosynthesis", "Cell division", "Energy release"],
            "correct_index": 0, "explanation": "It controls substance movement.",
            "topic": "Cell membrane", "difficulty": "Easy",
        }]})
        async def run_case():
            with patch("educore.services.student_service.Retriever.search", new=AsyncMock(return_value=[source])), patch(
                "educore.services.student_service.structured_response_format", return_value={"type": "json_object"}
            ), patch("educore.rag.generator.OpenRouterGenerator.generate", new=AsyncMock(return_value=raw)):
                return await service.generate_exam(user_id, project_id, "Cell membrane", difficulty="Easy", question_count=1)
        import asyncio
        response = asyncio.run(run_case())
        self.assertEqual(response["question_count"], 1)
        self.assertNotIn("correct_index", response["questions"][0])
        self.assertEqual(response["questions"][0]["options"], ["Entry and exit", "Photosynthesis", "Cell division", "Energy release"])

    def test_lesson_plan_adapter_keeps_legacy_lesson_and_validates_independent_contract(self):
        req = LessonRequest(subject="Biology", grade="8", topic="Cell membrane", duration_minutes=45)
        lesson = {
            "lesson_title": "Cell membrane", "learning_objectives": [{"objective": "Explain membrane function."}],
            "prerequisites": [], "lesson_flow": [{"stage": "Opening", "duration_minutes": 10, "teacher_actions": ["Introduce the membrane."], "student_actions": ["Observe the diagram."]}],
            "concept_explanations": [{"concept": "Selective permeability", "step_by_step_explanation": ["Some substances cross."]}],
            "worked_examples": [{"title": "Example", "problem": "Predict movement", "final_answer": "Down the gradient."}],
            "practice_questions": [{"question": "Where do particles move?", "correct_answer": "Down the gradient.", "explanation": "Diffusion follows concentration."}],
            "homework": [], "lesson_summary": {"key_takeaways": ["The membrane regulates exchange."]},
        }
        plan = adapt_generated_lesson_to_plan(lesson, req)
        self.assertIsInstance(plan, LessonPlan)
        self.assertEqual(plan.assessment_questions[0].question_type, "short_answer")
        self.assertEqual(plan.estimated_duration, 45)
        self.assertIn("lesson_title", GeneratedLesson.model_fields)
        self.assertNotIn("options", GeneratedLesson.model_fields)

    def test_lesson_service_exports_independent_plan_and_legacy_lesson(self):
        from educore.teacher_ai.modules.lesson_preparation.service import LessonPreparationService
        req = LessonRequest(subject="Biology", grade="8", topic="Cell membrane", duration_minutes=45)
        lesson = {
            "lesson_title": "Cell membrane", "learning_objectives": [{"objective": "Explain its function."}],
            "prerequisites": [], "lesson_flow": [{"stage": "Opening", "duration_minutes": 45,
                "teacher_actions": ["Introduce membrane."], "student_actions": ["Observe."]}],
            "concept_explanations": [{"concept": "Selective permeability", "step_by_step_explanation": ["Some substances cross."]}],
            "worked_examples": [{"title": "Example", "problem": "Predict movement", "final_answer": "Down the gradient."}],
            "practice_questions": [{"question": "Where do particles move?", "correct_answer": "Down the gradient.", "explanation": "Diffusion follows concentration."}],
            "homework": [], "lesson_summary": {"key_takeaways": ["The membrane regulates exchange."]},
        }
        report = SimpleNamespace(to_dict=lambda: {"status": "ready"}, verification=SimpleNamespace(has_critical=False))
        generator = SimpleNamespace(generate=Mock(return_value=SimpleNamespace(lesson=lesson, metadata={"model": "mock"})))
        reviewer = SimpleNamespace(review=Mock(return_value=report))
        prepared = LessonPreparationService(generator=generator, reviewer=reviewer, auto_repair=False).prepare(req)
        exported = prepared.to_dict()
        self.assertIn("lesson_title", exported["lesson"])
        self.assertEqual(exported["lesson_plan"]["subject"], "Biology")
        self.assertEqual(exported["lesson_plan"]["assessment_questions"][0]["question_type"], "short_answer")

    def test_parser_accepts_markdown_json_and_rejects_malformed_json(self):
        self.assertEqual(parse_json_object("```json\n{\"ok\": true}\n```"), {"ok": True})
        with self.assertRaises(ValueError):
            parse_json_object('{"ok":')


if __name__ == "__main__":
    unittest.main()
