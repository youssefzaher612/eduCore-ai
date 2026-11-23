import sys
import types
import unittest


# The tested report module imports the shared LLM client through its package
# initializer. These unit tests never call a provider, so provide only the SDK
# symbol needed during import when the optional runtime dependency is absent.
try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from pydantic import ValidationError

from educore.teacher_ai.modules.student_analytics.analyzer import analyze_student_performance
from educore.teacher_ai.modules.student_analytics.schemas import AnalyticsRequest
from educore.teacher_ai.modules.grading_feedback.schemas import GradingRequest
from educore.teacher_ai.modules.grading_feedback.grader import _reconcile_grading_data
from educore.teacher_ai.modules.reports_communication.schemas import ReportRequest
from educore.teacher_ai.modules.reports_communication.report_generator import generate_student_report
from educore.teacher_ai.modules.reports_communication import communication_generator
from educore.teacher_ai.modules.assessment_generator.schemas import (
    AssessmentRequest,
    AssessmentResult,
)
from educore.teacher_ai.modules.assessment_generator.ui import _build_assessment_request
from educore.teacher_ai.modules.lesson_preparation.schemas import LessonRequest
from educore.teacher_ai.modules.lesson_preparation.schemas import parse_generated_lesson
from educore.teacher_ai.modules.lesson_preparation.generator import (
    _apply_preparation_mode,
    _reconcile_lesson_timing,
)
from educore.rag.generator import _append_groq_search_citations
from educore.rag.retriever import RetrievalResult
from educore.rag.validator import EvidenceValidator
from unittest.mock import AsyncMock, Mock, patch
from educore.teacher_ai.core.llm_client import LLMClient


class TeacherModelFixTests(unittest.TestCase):
    def test_assessment_form_rejects_blank_subject_or_grade_without_validation_error(self):
        for subject, grade, expected_label in (
            ("   ", "5", "subject"),
            ("Mathematics", "  ", "grade"),
        ):
            with self.subTest(field=expected_label):
                request, message = _build_assessment_request(subject, grade)
                self.assertIsNone(request)
                self.assertIn(expected_label, message)

    def test_assessment_form_strips_subject_and_grade(self):
        request, message = _build_assessment_request(
            " Mathematics ",
            " Grade 5 ",
            topic="Fractions",
            assessment_type="quiz",
            difficulty="easy",
            question_types=["mcq"],
            question_count=1,
            total_marks=2,
            duration_minutes=10,
            language="English",
        )
        self.assertIsNone(message)
        self.assertEqual(request.subject, "Mathematics")
        self.assertEqual(request.grade, "Grade 5")

    def test_latest_assessment_uses_date_not_input_order(self):
        request = AnalyticsRequest(
            subject="Math",
            grade_level="5",
            records=[
                {
                    "student_id": "s1", "student_name": "A", "assessment_title": "new",
                    "subject": "Math", "grade_level": "5", "awarded_marks": 90,
                    "max_marks": 100, "assessment_date": "2026-02-01",
                },
                {
                    "student_id": "s1", "student_name": "A", "assessment_title": "old",
                    "subject": "Math", "grade_level": "5", "awarded_marks": 10,
                    "max_marks": 100, "assessment_date": "2026-01-01",
                },
            ],
        )
        result = analyze_student_performance(request)
        self.assertEqual(result.students[0].latest_percentage, 90.0)

    def test_invalid_assessment_date_is_rejected(self):
        with self.assertRaises(ValidationError):
            AnalyticsRequest(
                subject="Math", grade_level="5", records=[{
                    "student_id": "s1", "student_name": "A", "assessment_title": "x",
                    "subject": "Math", "grade_level": "5", "awarded_marks": 1,
                    "max_marks": 2, "assessment_date": "not-a-date",
                }],
            )

    def test_student_average_is_weighted_by_marks(self):
        request = AnalyticsRequest(
            subject="Math", grade_level="5", records=[
                {
                    "student_id": "s1", "student_name": "A", "assessment_title": "short",
                    "subject": "Math", "grade_level": "5", "awarded_marks": 1,
                    "max_marks": 1,
                },
                {
                    "student_id": "s1", "student_name": "A", "assessment_title": "major",
                    "subject": "Math", "grade_level": "5", "awarded_marks": 0,
                    "max_marks": 9,
                },
            ],
        )
        result = analyze_student_performance(request)
        self.assertEqual(result.students[0].average_percentage, 10.0)

    def test_grading_rejects_duplicate_answers_and_unknown_ids(self):
        question = {
            "question_id": "q1", "question_text": "2+2?", "question_type": "short_answer",
            "max_marks": 1,
        }
        common = {"subject": "Math", "grade_level": "5", "questions": [question]}
        with self.assertRaises(ValidationError):
            GradingRequest(**common, student_answers=[
                {"question_id": "q1", "answer": "4"},
                {"question_id": "q1", "answer": "four"},
            ])
        with self.assertRaises(ValidationError):
            GradingRequest(**common, student_answers=[{"question_id": "q2", "answer": "4"}])

    def test_exact_expected_answer_receives_full_credit(self):
        request = GradingRequest(
            subject="Math", grade_level="5", questions=[{
                "question_id": "q1", "question_text": "What is 2 + 3?",
                "question_type": "short_answer", "max_marks": 2, "expected_answer": "5",
            }], student_answers=[{"question_id": "q1", "answer": " 5 "}],
        )
        data = {"question_grades": [{
            "question_id": "q1", "awarded_marks": 1, "is_fully_correct": False,
            "feedback": "Partially correct", "criterion_grades": [],
        }]}
        result = _reconcile_grading_data(request, data)
        self.assertEqual(result["question_grades"][0]["awarded_marks"], 2)
        self.assertTrue(result["question_grades"][0]["is_fully_correct"])
        self.assertEqual(result["percentage"], 100)

    def test_student_report_requires_student_id(self):
        request = ReportRequest(report_title="Report", subject="Math", grade_level="5")
        with self.assertRaisesRegex(ValueError, "student_id is required"):
            generate_student_report(request, types.SimpleNamespace(students=[]))

    def test_communication_rejects_missing_required_output(self):
        request = {
            "recipient_type": "parent", "recipient_name": "Parent",
            "student_name": "Student", "subject": "Math", "grade_level": "5",
            "purpose": "Progress update",
        }
        with patch.object(communication_generator.llm_client, "generate", return_value='{"subject_line":""}'):
            with self.assertRaisesRegex(ValueError, "invalid communication message"):
                communication_generator.generate_communication(request)

    def test_assessment_request_and_result_constraints(self):
        with self.assertRaises(ValidationError):
            AssessmentRequest(
                subject="Math", grade="5", topic="Algebra", assessment_type="quiz",
                difficulty="easy", question_types=["mcq"], question_count=0,
                total_marks=10, duration_minutes=30,
            )
        with self.assertRaises(ValidationError):
            AssessmentResult(
                title="Quiz", questions=[{
                    "question_number": 1, "question_type": "mcq", "difficulty": "easy",
                    "question_text": "2+2?", "marks": 1,
                    "options": [{"id": "a", "text": "4"}],
                }], total_marks=1, duration_minutes=10,
            )

    def test_lesson_request_rejects_out_of_range_duration(self):
        with self.assertRaises(ValidationError):
            LessonRequest(subject="Math", grade="5", topic="Fractions", duration_minutes=5)

    def test_lesson_normalizes_structured_practice_steps(self):
        lesson = parse_generated_lesson({
            "lesson_title": "Adding fractions",
            "homework": [{
                "question": "Add 2/5 + 1/5.", "difficulty": "Easy",
                "answer_or_guidance": "3/5",
            }],
            "practice_questions": [{
                "question": "Add 1/5 + 2/5.", "difficulty": "Basic",
                "step_by_step_solution": [{"step": "Add numerators", "work": "1 + 2 = 3"}],
                "correct_answer": "3/5", "explanation": "Keep the denominator.",
            }],
        })
        self.assertEqual(
            lesson["practice_questions"][0]["step_by_step_solution"],
            ["Add numerators — 1 + 2 = 3"],
        )
        self.assertEqual(lesson["homework"][0]["difficulty"], "Basic")

    def test_prepare_mode_clears_teacher_learning_extras(self):
        request = LessonRequest(subject="Math", grade="5", topic="Fractions")
        lesson = {"teacher_learning": {
            "include": True, "foundations": [{"concept": "x"}],
            "teacher_examples": ["example"], "knowledge_warnings": ["warning"],
        }}
        _apply_preparation_mode(lesson, request)
        self.assertEqual(lesson["teacher_learning"], {
            "include": False, "foundations": [], "teacher_examples": [], "knowledge_warnings": [],
        })

    def test_lesson_timing_reconciles_without_stopping_above_target(self):
        stages = [
            {"duration_minutes": 8},
            {"duration_minutes": 4},
            {"duration_minutes": 5},
        ]
        _reconcile_lesson_timing(stages, 15)
        self.assertEqual(sum(stage["duration_minutes"] for stage in stages), 15)
        self.assertTrue(all(stage["duration_minutes"] >= 1 for stage in stages))

    def test_groq_browser_results_become_markdown_citations(self):
        message = types.SimpleNamespace(executed_tools=[{
            "search_results": {"results": [
                {"title": "Official source", "url": "https://example.org/page"},
                {"title": "Unsafe URL", "url": "javascript:alert(1)"},
            ]},
        }])
        answer = _append_groq_search_citations(message, "Answer")
        self.assertIn("[Official source](https://example.org/page)", answer)
        self.assertNotIn("javascript:", answer)


class GroqEvidenceFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_validator_uses_local_evidence_without_provider_call(self):
        source = RetrievalResult(
            document_id="doc1", document_name="lecture.txt", page_number=1,
            content="Natural language processing enables computers to work with human language.",
            similarity=0.9,
            chunk_id="chunk1",
        )
        with patch("educore.rag.validator.create_async_client", new_callable=AsyncMock) as create:
            status = await EvidenceValidator("openai/gpt-oss-120b").validate(
                "What does NLP do?", [source]
            )
        self.assertEqual(status, "SUPPORTED")
        create.assert_not_awaited()


class TeacherLLMCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _completion(content='{"ok":true}'):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content),
                finish_reason="stop",
            )],
            usage=types.SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            model="openai/gpt-oss-20b",
            id="test-response",
        )

    def test_json_validate_failed_retries_same_model_without_response_format(self):
        rejected = Exception("HTTP 400")
        rejected.status_code = 400
        rejected.body = {
            "error": {
                "code": "json_validate_failed",
                "message": "The constrained JSON generation failed.",
            }
        }
        create = Mock(side_effect=[rejected, self._completion()])
        sdk = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        ))
        client = LLMClient(sdk_client=sdk, model="openai/gpt-oss-20b")

        response = client.generate_detailed("Return JSON.", "Return {\"ok\":true}.", max_tokens=300)

        self.assertEqual(response.content, '{"ok":true}')
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0].kwargs["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", create.call_args_list[1].kwargs)
        self.assertEqual(create.call_args_list[0].kwargs["model"], create.call_args_list[1].kwargs["model"])

    def test_recent_primary_rate_limit_still_tries_fast_groq_model(self):
        previous_block = LLMClient._groq_blocked_until
        self.addCleanup(setattr, LLMClient, "_groq_blocked_until", previous_block)
        LLMClient._groq_blocked_until = 9999999999
        create = Mock(return_value=self._completion())
        sdk = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        ))
        client = LLMClient(sdk_client=sdk, model="openai/gpt-oss-120b")

        with patch("educore.teacher_ai.core.llm_client.settings.AI_PROVIDER", "groq"), patch("educore.teacher_ai.core.llm_client.settings.LLM_PROVIDER", "groq"), patch("educore.teacher_ai.core.llm_client._FALLBACK_MODELS", ["openai/gpt-oss-20b"]):
            response = client.generate_detailed("Return JSON.", "Return {\"ok\":true}.", max_tokens=300)

        self.assertEqual(response.content, '{"ok":true}')
        self.assertEqual(create.call_args.kwargs["model"], "openai/gpt-oss-20b")

    def test_primary_rate_limit_then_fast_json_rejection_recovers(self):
        previous_block = LLMClient._groq_blocked_until
        self.addCleanup(setattr, LLMClient, "_groq_blocked_until", previous_block)
        LLMClient._groq_blocked_until = 0
        rate_limited = Exception("HTTP 429")
        rate_limited.status_code = 429
        rejected = Exception("HTTP 400")
        rejected.status_code = 400
        rejected.body = {"error": {"code": "json_validate_failed"}}
        create = Mock(side_effect=[rate_limited, rejected, self._completion()])
        sdk = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        ))
        client = LLMClient(sdk_client=sdk, model="openai/gpt-oss-120b", sleep=lambda _: None)

        with patch("educore.teacher_ai.core.llm_client.settings.AI_PROVIDER", "groq"), patch("educore.teacher_ai.core.llm_client.settings.LLM_PROVIDER", "groq"), patch("educore.teacher_ai.core.llm_client._FALLBACK_MODELS", ["openai/gpt-oss-20b"]):
            response = client.generate_detailed("Return JSON.", "Return {\"ok\":true}.", max_tokens=300)

        self.assertEqual(response.content, '{"ok":true}')
        self.assertEqual([call.kwargs["model"] for call in create.call_args_list], [
            "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-20b",
        ])
        self.assertIn("response_format", create.call_args_list[1].kwargs)
        self.assertNotIn("response_format", create.call_args_list[2].kwargs)


if __name__ == "__main__":
    unittest.main()

