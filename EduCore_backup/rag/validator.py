
"""
Strict evidence validator for EduCore RAG.

Uses token-based exact sequence matching to handle PDF extraction
formatting issues without allowing paraphrased evidence.
"""

from __future__ import annotations

import json
import logging
import re

from groq import AsyncGroq

from educore.config import settings

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase alphanumeric tokens.

    Handles:
    - Uppercase/lowercase differences
    - Line breaks and repeated spaces
    - Missing spaces between extracted PDF words
    - Punctuation differences
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def contains_exact_token_sequence(
    quote: str,
    source: str,
) -> bool:
    """
    Check whether all quote tokens appear consecutively
    and in the same order inside the source.
    """
    quote_tokens = tokenize(quote)
    source_tokens = tokenize(source)

    if not quote_tokens or len(quote_tokens) > len(source_tokens):
        return False

    quote_length = len(quote_tokens)

    for index in range(
        len(source_tokens) - quote_length + 1
    ):
        if source_tokens[index:index + quote_length] == quote_tokens:
            return True

    return False


class EvidenceValidator:

    def __init__(self):
        self.client = AsyncGroq(
            api_key=settings.GROQ_API_KEY
        )

    async def validate(
        self,
        question: str,
        sources: list,
    ) -> str:

        if not sources:
            logger.warning("No sources available for validation")
            return "NOT_SUPPORTED"

        source_text = "\n\n".join(
            f"[{i}] {source.content}"
            for i, source in enumerate(sources, start=1)
        )

        prompt = f"""
You are a strict evidence verification system.

Determine whether the supplied sources explicitly contain
enough information to answer the question.

RULES:
1. Use ONLY the supplied sources.
2. Never use general knowledge.
3. A broad mention of a topic is not enough.
4. Do not infer missing definitions.
5. Every supported claim must have direct evidence.
6. If the concept is not explained, return NOT_SUPPORTED.
7. Similarity between the question and source topic is not evidence.
8. Copy quotes from the source without paraphrasing.
9. Preserve the original words and their order.
10. You may ignore line breaks when copying a quote.
11. Select the source that directly explains the requested concept.

QUESTION:
{question}

SOURCES:
{source_text}

Return valid JSON only:

{{
  "status": "SUPPORTED",
  "evidence": [
    {{
      "claim": "claim being verified",
      "source_number": 1,
      "quote": "words copied from the source"
    }}
  ]
}}

Allowed statuses:
SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED

For NOT_SUPPORTED, evidence must be an empty list.

For PARTIALLY_SUPPORTED, include only directly supported claims.

For SUPPORTED, every essential part of the question
must have supporting evidence.
"""

        try:
            response = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evidence verifier. "
                            "Return valid JSON only. "
                            "Never use external knowledge."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=500,
            )

            content = (
                response.choices[0].message.content or ""
            ).strip()

            logger.info(
                "RAW VALIDATOR RESPONSE: %s",
                content,
            )

            # Remove optional Markdown code fences.
            if content.startswith("```"):
                content = re.sub(
                    r"^```(?:json)?\s*",
                    "",
                    content,
                )
                content = re.sub(
                    r"\s*```$",
                    "",
                    content,
                )

            result = json.loads(content)

            status = result.get("status")
            evidence = result.get("evidence", [])

            if status not in {
                "SUPPORTED",
                "PARTIALLY_SUPPORTED",
                "NOT_SUPPORTED",
            }:
                logger.warning(
                    "Invalid validator status: %r",
                    status,
                )
                return "NOT_SUPPORTED"

            if not isinstance(evidence, list):
                logger.warning("Evidence must be a list")
                return "NOT_SUPPORTED"

            if status == "NOT_SUPPORTED":
                logger.info(
                    "EVIDENCE VALIDATION: NOT_SUPPORTED"
                )
                return "NOT_SUPPORTED"

            verified_evidence = []

            for item in evidence:

                if not isinstance(item, dict):
                    logger.warning(
                        "Invalid evidence item: %r",
                        item,
                    )
                    continue

                quote = item.get("quote")
                source_number = item.get("source_number")

                logger.info(
                    "VALIDATING EVIDENCE: %r",
                    item,
                )

                if (
                    not isinstance(quote, str)
                    or not quote.strip()
                    or not isinstance(source_number, int)
                    or isinstance(source_number, bool)
                    or source_number < 1
                    or source_number > len(sources)
                ):
                    logger.warning(
                        "Invalid evidence reference: %r",
                        item,
                    )
                    continue

                actual_source = sources[
                    source_number - 1
                ].content

                # Compare ordered tokens instead of raw characters.
                if contains_exact_token_sequence(
                    quote,
                    actual_source,
                ):
                    verified_evidence.append(item)

                    logger.info(
                        "QUOTE VERIFIED | SOURCE=%d | QUOTE=%r",
                        source_number,
                        quote,
                    )

                else:
                    logger.warning(
                        "QUOTE VERIFICATION FAILED | "
                        "SOURCE=%d | QUOTE=%r",
                        source_number,
                        quote,
                    )

            logger.info(
                "VALIDATOR STATUS=%s | "
                "TOTAL_EVIDENCE=%d | VERIFIED_QUOTES=%d",
                status,
                len(evidence),
                len(verified_evidence),
            )

            if not verified_evidence:
                return "NOT_SUPPORTED"

            if status == "SUPPORTED":

                if len(verified_evidence) != len(evidence):
                    return "NOT_SUPPORTED"

                logger.info(
                    "EVIDENCE VALIDATION: SUPPORTED"
                )
                return "SUPPORTED"

            if status == "PARTIALLY_SUPPORTED":
                logger.info(
                    "EVIDENCE VALIDATION: PARTIALLY_SUPPORTED"
                )
                return "PARTIALLY_SUPPORTED"

            return "NOT_SUPPORTED"

        except json.JSONDecodeError:
            logger.exception(
                "Validator returned invalid JSON"
            )
            return "NOT_SUPPORTED"

        except Exception:
            logger.exception(
                "Evidence validation failed unexpectedly"
            )
            return "NOT_SUPPORTED"
