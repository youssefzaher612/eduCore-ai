import unittest
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from educore.config import settings
from educore.provider import provider_settings, require_api_key
from educore.rag.generator import OpenRouterGenerator
from educore.rag.response_cache import ExactResponseCache
from educore.rag.response_cache import SemanticResponseCache
from educore.rag.extractor import DocumentExtractor, ExtractedPage
from educore.teacher_ai.core.llm_client import LLMClient
from educore.teacher_ai.core.llm_client import LLMRequestError
from educore.teacher_ai.modules.lesson_preparation.generator import LessonGenerator
from educore.teacher_ai.modules.lesson_preparation.prompts import (
    LOCAL_SYSTEM_PROMPT, build_local_lesson_prompt,
)
from educore.teacher_ai.modules.lesson_preparation.schemas import LessonRequest
from educore.services.student_service import StudentLearningService
from educore.student_ai_schemas import StudentExamOutput


class LocalProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_student_exam_generation_uses_bounded_question_scaled_token_budget(self):
        self.assertEqual(StudentLearningService._exam_output_token_budget(1), 420)
        self.assertEqual(StudentLearningService._exam_output_token_budget(5), 1300)
        self.assertEqual(StudentLearningService._exam_output_token_budget(20), 4096)

    def test_student_exam_fills_missing_topic_and_difficulty_without_retry(self):
        payload = StudentExamOutput.model_validate({
            "title": "Practice",
            "questions": [{
                "question": "What is overfitting?",
                "options": ["Memorizing examples", "More data", "Faster training", "No features"],
                "correct_index": 0,
                "explanation": "It memorizes training examples and generalizes poorly.",
            }],
        })
        question = StudentLearningService._validate_generated_questions(
            [item.model_dump() for item in payload.questions],
            "Overfitting", "Easy", 1, {"Overfitting"},
        )[0]
        self.assertEqual(question["topic"], "Overfitting")
        self.assertEqual(question["difficulty"], "Easy")

    def test_local_lesson_prompt_fits_small_context_and_keeps_all_sections(self):
        request = LessonRequest(
            subject="Biology", grade="Secondary 2", topic="Cell membrane",
            language="Arabic", question_count=3, duration_minutes=45,
        )
        prompt = build_local_lesson_prompt(
            request, "The membrane regulates movement into and out of the cell. " * 40
        )
        self.assertLess(len(LOCAL_SYSTEM_PROMPT) + len(prompt), 6000)
        for section in (
            "lesson_title", "lesson_overview", "lesson_flow", "concept_explanations",
            "worked_examples", "practice_questions", "lesson_summary", "differentiation",
        ):
            self.assertIn(section, prompt)

    def test_local_lesson_retries_context_overflow_with_compact_rag_source(self):
        request = LessonRequest(
            subject="Biology", grade="Grade 8", topic="Cells",
            language="English", question_count=1, duration_minutes=15,
        )
        lesson = {
            "lesson_title": "Cells",
            "lesson_overview": {"central_idea": "Cells are life's basic units.", "expected_outcome": "Name key parts."},
            "learning_objectives": [{"objective": "Identify cell parts."}],
            "prerequisites": [],
            "teacher_learning": {"include": False},
            "teaching_strategy": {"recommended_approach": "Model then guided practice."},
            "lesson_flow": [{"stage": "Teach", "duration_minutes": 15, "teacher_actions": ["Explain."], "student_actions": ["Label a diagram."]}],
            "concept_explanations": [{"concept": "Cell", "step_by_step_explanation": [{"step": "1", "teacher_explanation": "A cell is a basic unit."}]}],
            "worked_examples": [{"title": "Label a cell", "difficulty": "Basic", "problem": "Identify the membrane.", "solution_steps": [{"work": "Find the outer boundary."}], "final_answer": "The outer boundary is the membrane."}],
            "classroom_interaction": [], "classroom_activities": [], "common_misconceptions": [], "formative_assessment": [],
            "practice_questions": [{"question": "Name one cell part.", "difficulty": "Basic", "step_by_step_solution": ["Recall the diagram."], "correct_answer": "Any correctly named part.", "explanation": "Cells contain specialized parts."}],
            "lesson_summary": {"key_takeaways": ["Cells have specialized parts."]}, "homework": [],
            "differentiation": {"students_at_expected_level": ["Label the diagram."]},
            "teacher_notes": [], "post_lesson_reflection": [],
        }
        fake_client = SimpleNamespace(generate_detailed=Mock(side_effect=[
            LLMRequestError("HTTP 400", status_code=400, detail="prompt is longer than the context length"),
            SimpleNamespace(content=json.dumps(lesson), to_metadata=lambda: {}),
        ]))
        with patch.object(settings, "AI_PROVIDER", ""), patch.object(
            settings, "LLM_PROVIDER", "ollama"
        ):
            result = LessonGenerator(client=fake_client).generate(
                request, source_material="cell source " * 500
            )
        self.assertEqual(result.lesson["lesson_title"], "Cells")
        self.assertEqual(fake_client.generate_detailed.call_count, 2)
        first_call, second_call = fake_client.generate_detailed.call_args_list
        self.assertEqual(first_call.args[0], LOCAL_SYSTEM_PROMPT)
        self.assertLess(len(second_call.args[1]), len(first_call.args[1]))

    def test_ollama_needs_no_secret_and_uses_openai_compatibility_url(self):
        with patch.object(settings, "AI_PROVIDER", ""), patch.object(
            settings, "LLM_PROVIDER", "ollama"
        ), patch.object(settings, "OLLAMA_BASE_URL", "http://localhost:11434/"):
            provider, key, base_url = provider_settings()
            self.assertEqual(provider, "ollama")
            self.assertEqual(key, "ollama")
            self.assertEqual(base_url, "http://localhost:11434/v1")
            self.assertEqual(require_api_key(), "ollama")

    async def test_local_generation_uses_max_tokens_and_bounded_context(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))],
            usage=SimpleNamespace(total_tokens=4),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=response))
            )
        )
        with patch.object(settings, "AI_PROVIDER", ""), patch.object(
            settings, "LLM_PROVIDER", "ollama"
        ), patch(
            "educore.rag.generator.create_async_client", return_value=client
        ):
            answer = await OpenRouterGenerator(model_id="qwen3-vl:2b").generate(
                [{"role": "user", "content": "Reply ready"}], max_tokens=256
            )

        self.assertEqual(answer, "ready")
        request = client.chat.completions.create.await_args.kwargs
        self.assertEqual(request["max_tokens"], 256)
        self.assertNotIn("max_completion_tokens", request)
        self.assertEqual(request["extra_body"]["options"]["num_ctx"], settings.OLLAMA_NUM_CTX)
        self.assertEqual(request["extra_body"]["keep_alive"], settings.OLLAMA_KEEP_ALIVE)
        self.assertNotIn("reasoning_effort", request)

    def test_teacher_json_generation_uses_local_token_and_context_settings(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"ok":true}'), finish_reason="stop"
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=4, total_tokens=9),
            model="qwen3-vl:2b",
            id="local-test",
        )
        create = Mock(return_value=response)
        sdk = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        ))
        with patch.object(settings, "AI_PROVIDER", ""), patch.object(
            settings, "LLM_PROVIDER", "ollama"
        ):
            client = LLMClient(
                sdk_client=sdk, model="qwen3-vl:2b", json_mode=True,
                max_retries=0, model_max_completion_tokens=128,
            )
            result = client.generate_detailed(
                "Return valid JSON.", "Return an object with ok true.", max_tokens=64
            )

        self.assertEqual(result.content, '{"ok":true}')
        request = create.call_args.kwargs
        self.assertEqual(request["max_tokens"], 64)
        self.assertNotIn("max_completion_tokens", request)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"]["options"]["num_ctx"], settings.OLLAMA_NUM_CTX)
        self.assertEqual(request["extra_body"]["keep_alive"], settings.OLLAMA_KEEP_ALIVE)

    def test_exact_cache_is_scoped_by_user_project_role_and_task(self):
        cache = ExactResponseCache(max_entries=2, ttl_seconds=60)
        base = {
            "user_id": "user-a",
            "project_id": "project-a",
            "role": "student",
            "task": "rag:study:auto",
            "model": "qwen3-vl:2b",
            "messages": [{"role": "user", "content": "Explain this"}],
        }
        key = cache.key(**base)
        cache.put(key, "grounded answer")
        self.assertEqual(cache.get(key), "grounded answer")
        self.assertNotEqual(
            key,
            cache.key(**{**base, "user_id": "user-b"}),
        )
        self.assertNotEqual(
            key,
            cache.key(**{**base, "project_id": "project-b"}),
        )

    def test_semantic_cache_requires_same_private_context_and_near_duplicate(self):
        cache = SemanticResponseCache(threshold=0.96)
        base = {
            "user_id": "user-a", "project_id": "project-a", "role": "student",
            "task": "rag:study:auto", "model": "qwen3-vl:2b",
            "messages": [
                {"role": "system", "content": "Lecture chunk: overfitting means..."},
                {"role": "user", "content": "Explain overfitting"},
            ],
        }
        scope, query = cache.scope_key(**base)
        self.assertEqual(query, "Explain overfitting")
        cache.put(scope, [1.0, 0.0], "grounded answer")
        self.assertEqual(cache.get(scope, [0.999, 0.04]), "grounded answer")
        self.assertIsNone(cache.get(scope, [0.0, 1.0]))
        other_scope, _ = cache.scope_key(**{**base, "user_id": "user-b"})
        self.assertIsNone(cache.get(other_scope, [1.0, 0.0]))

    def test_document_extraction_cache_uses_scope_and_content_version(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as source:
            path = Path(source.name)
            source.write(b"first version")
        extracted = [ExtractedPage("cached page", 1, {"language": "en"})]
        try:
            with patch.object(DocumentExtractor, "_extract_text", return_value=extracted) as handler:
                DocumentExtractor.extract(str(path), "txt", cache_scope="project-a/doc-a")
                DocumentExtractor.extract(str(path), "txt", cache_scope="project-a/doc-a")
                self.assertEqual(handler.call_count, 1)
                path.write_bytes(b"updated version")
                DocumentExtractor.extract(str(path), "txt", cache_scope="project-a/doc-a")
                self.assertEqual(handler.call_count, 2)
                DocumentExtractor.extract(str(path), "txt", cache_scope="project-b/doc-a")
                self.assertEqual(handler.call_count, 3)
        finally:
            path.unlink(missing_ok=True)
