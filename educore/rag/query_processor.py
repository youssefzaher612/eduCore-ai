
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedQuery:
    original_question: str
    retrieval_query: str
    requested_language: str | None
    answer_instructions: str
    requested_sentence_count: int | None = None


_LANGUAGE_PATTERNS = {
    "English": [
        r"\banswer\s+in\s+english\b",
        r"\brespond\s+in\s+english\b",
        r"\breply\s+in\s+english\b",
        r"\benglish\s+only\b",
        r"\buse\s+english\b",
    ],
    "Arabic": [
        r"\banswer\s+in\s+arabic\b",
        r"\brespond\s+in\s+arabic\b",
        r"\breply\s+in\s+arabic\b",
        r"\barabic\s+only\b",
        r"\buse\s+arabic\b",
        r"بالعربي\s*فقط",
        r"جاوب\s+بالعربي",
        r"جاوب\s+باللغة\s+العربية",
        r"أجب\s+بالعربية",
        r"جاوب\s+عربي",
    ],
}

_INSTRUCTION_PATTERNS = [
    r"\banswer\s+briefly\b",
    r"\banswer\s+in\s+detail\b",
    r"\bexplain\s+in\s+detail\b",
    r"\bstep\s+by\s+step\b",
    r"\buse\s+bullet\s+points\b",
    r"\bmake\s+it\s+short\b",
    r"\bbriefly\b",
    r"\bin\s+detail\b",
    r"باختصار",
    r"بالتفصيل",
    r"خطوة\s+بخطوة",
    r"اشرح\s+بالتفصيل",
    r"اشرح\s+باختصار",
]

_ARABIC_INSTRUCTION_PATTERNS = [
    r"و?\s*جاوب\s+بالعربي",
    r"و?\s*جاوب\s+باللغة\s+العربية",
    r"و?\s*جاوب\s+عربي",
    r"و?\s*أجب\s+بالعربية",
    r"و?\s*اشرح\s+بالتفصيل",
    r"و?\s*اشرح\s+باختصار",
]

_SENTENCE_COUNT_PATTERNS = [
    r"\b(?:in|within)\s+(?P<count>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?\b",
    r"\b(?P<count>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?\b",
    r"(?:في|خلال|ب)\s*(?P<count>[٠-٩0-9]{1,2})\s*جمل(?:ة)?",
    r"(?:في|خلال|ب)\s*(?P<count>جملة|جمله|جملتين|ثلاث(?:ة)?\s+جمل|أربع(?:ة)?\s+جمل|خمس\s+جمل|ست\s+جمل|سبع\s+جمل|ثماني\s+جمل|تسع\s+جمل|عشر\s+جمل)",
]

_SENTENCE_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "جملة": 1,
    "جمله": 1,
    "جملتين": 2,
    "ثلاث جمل": 3,
    "ثلاثة جمل": 3,
    "أربع جمل": 4,
    "أربعة جمل": 4,
    "خمس جمل": 5,
    "ست جمل": 6,
    "سبع جمل": 7,
    "ثماني جمل": 8,
    "تسع جمل": 9,
    "عشر جمل": 10,
}


def _requested_sentence_count(question: str) -> int | None:
    normalized = question.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    for pattern in _SENTENCE_COUNT_PATTERNS:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if not match:
            continue
        raw_count = match.group("count").strip().casefold()
        count = int(raw_count) if raw_count.isdigit() else _SENTENCE_COUNT_WORDS.get(raw_count)
        if count is not None:
            return max(1, min(count, 12))
    return None


def process_query(question: str) -> ProcessedQuery:
    original = (question or "").strip()

    if not original:
        return ProcessedQuery("", "", None, "")

    requested_sentence_count = _requested_sentence_count(original)

    # Detect requested answer language.
    requested_language = None

    for language, patterns in _LANGUAGE_PATTERNS.items():
        if any(
            re.search(pattern, original, re.IGNORECASE)
            for pattern in patterns
        ):
            requested_language = language
            break

    # Detect whether the user requested a specific answer format or length.
    has_format_instruction = any(
        re.search(pattern, original, re.IGNORECASE)
        for pattern in _INSTRUCTION_PATTERNS
    ) or any(
        re.search(pattern, original, re.IGNORECASE)
        for pattern in _ARABIC_INSTRUCTION_PATTERNS
    ) or requested_sentence_count is not None

    # Remove language and answer-format instructions from retrieval text.
    retrieval_query = original

    all_patterns = (
        [
            pattern
            for patterns in _LANGUAGE_PATTERNS.values()
            for pattern in patterns
        ]
        + _INSTRUCTION_PATTERNS
        + _ARABIC_INSTRUCTION_PATTERNS
        + _SENTENCE_COUNT_PATTERNS
    )

    for pattern in all_patterns:
        retrieval_query = re.sub(
            pattern,
            " ",
            retrieval_query,
            flags=re.IGNORECASE,
        )

    # Remove leftover English conjunctions and Arabic conjunctions.
    retrieval_query = re.sub(
        r"\b(and|or|then)\b",
        " ",
        retrieval_query,
        flags=re.IGNORECASE,
    )

    retrieval_query = re.sub(
        r"\s+و\s*$",
        "",
        retrieval_query,
    )

    # Normalize whitespace and punctuation.
    retrieval_query = re.sub(
        r"[\s,;:،؛\-]+",
        " ",
        retrieval_query,
    ).strip(" .?!؟")

    retrieval_query = retrieval_query.strip()

    # Keep the original question if processing removes too much text.
    if len(retrieval_query) < 3:
        retrieval_query = original

    # Build answer instructions separately from retrieval text.
    instruction_parts = []

    if requested_language:
        instruction_parts.append(
            f"Answer entirely in {requested_language}."
        )

    if has_format_instruction:
        instruction_parts.append(
            "Follow the requested answer length, detail, or format."
        )

    if requested_sentence_count is not None:
        instruction_parts.append(
            f"Answer in exactly {requested_sentence_count} sentences. "
            "Do not add a preface, heading, or separate source list."
        )

    return ProcessedQuery(
        original_question=original,
        retrieval_query=retrieval_query,
        requested_language=requested_language,
        answer_instructions=" ".join(instruction_parts),
        requested_sentence_count=requested_sentence_count,
    )
