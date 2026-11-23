import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from educore.config import settings
from educore.rag.generator import OpenRouterGenerator, _groq_reasoning_effort
from educore.rag.validator import extract_sentences


class RagPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_groq_generator_does_not_send_reasoning_effort(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))],
            usage=SimpleNamespace(total_tokens=4),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=response))
            )
        )
        with (
            patch.object(settings, "LLM_PROVIDER", "groq"),
            patch("educore.rag.generator.create_async_client", return_value=client),
        ):
            answer = await OpenRouterGenerator(
                model_id="groq/llama-3.1-8b-instant", reasoning_effort="medium"
            ).generate([{"role": "user", "content": "Reply ready"}], reasoning_effort="none")

        self.assertEqual(answer, "ready")
        request = client.chat.completions.create.await_args.kwargs
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(request["model"], "groq/llama-3.1-8b-instant")

    def test_groq_reasoning_effort_never_sends_unsupported_none(self):
        self.assertEqual(_groq_reasoning_effort("none", "medium"), "low")
        self.assertEqual(_groq_reasoning_effort("high", "low"), "high")

    def test_validator_splits_arabic_sentence_terminators(self):
        sentences = extract_sentences("الشمس مصدر للطاقة. النباتات تصنع الغذاء؟ الماء مهم۔")
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[1].endswith("؟"))
        self.assertTrue(sentences[2].endswith("۔"))

    async def test_malformed_supported_status_fails_closed(self):
        from educore.rag.validator import EvidenceValidator

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"status":"SUPPORTED","evidence":['
            ))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=response))
            )
        )
        with patch("educore.rag.validator.create_async_client", return_value=client):
            validator = EvidenceValidator(model_id="test-model")
            status = await validator.validate(
                "What does the source explain?",
                [SimpleNamespace(content="Photosynthesis converts light energy into chemical energy.")],
            )

        self.assertEqual(status, "NOT_SUPPORTED")
