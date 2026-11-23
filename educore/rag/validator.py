"""Local citation-reference validation for retrieved RAG chunks."""

from __future__ import annotations

import re

# Retained as patchable module attributes for older integrations. The local
# EvidenceValidator below never reads settings or creates a provider client.
from educore.config import settings
from educore.provider import create_async_client

_CHUNK_REFERENCE = re.compile(r"\[chunk:([^\]\s]+)\]", re.IGNORECASE)
_CHUNK_MARKER = re.compile(r"\[chunk[^\]]*\]", re.IGNORECASE)
_WORD = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_QUESTION_FILLER = {
    "a", "an", "and", "are", "does", "explain", "for", "how", "is", "of",
    "please", "source", "tell", "the", "this", "to", "what", "which", "who",
    "why", "ن", "ما", "ماذا", "كيف", "هل", "في", "عن", "هو", "هي",
}


def extract_sentences(text: str) -> list[str]:
    """Split text at common English and Arabic sentence terminators."""
    if not text or not text.strip():
        return []
    normalized = re.sub(r"\s+", " ", text).strip()
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?؟۔])\s+", normalized)
        if len(part.strip()) >= 8
    ]


def validate_chunk_citations(answer: str, sources: list) -> tuple[str, list[dict], bool]:
    """Keep only references to chunks actually supplied to the answer model.

    The model's answer is preserved when references are missing or invalid. The
    returned boolean signals that the user should be told citations could not
    all be matched.
    """
    source_by_id = {
        str(source.chunk_id).casefold(): (index, source)
        for index, source in enumerate(sources, start=1)
        if getattr(source, "chunk_id", None)
    }
    matches = list(_CHUNK_MARKER.finditer(answer or ""))
    referenced_ids = []
    invalid_reference = False

    def replace_reference(match: re.Match) -> str:
        nonlocal invalid_reference
        exact_reference = _CHUNK_REFERENCE.fullmatch(match.group(0))
        if exact_reference is None:
            invalid_reference = True
            return ""
        chunk_id = exact_reference.group(1)
        source_entry = source_by_id.get(chunk_id.casefold())
        if source_entry is None:
            invalid_reference = True
            return ""
        if chunk_id not in referenced_ids:
            referenced_ids.append(chunk_id)
        source_number, _ = source_entry
        return f" \u2066[{source_number}]\u2069"

    cleaned = _CHUNK_MARKER.sub(replace_reference, answer or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.?!؟،؛])", r"\1", cleaned).strip()

    citation_issue = invalid_reference or not matches or not referenced_ids
    citations = []
    referenced_set = set(referenced_ids)
    for index, source in enumerate(sources, start=1):
        chunk_id = str(getattr(source, "chunk_id", "") or "")
        if chunk_id.casefold() not in {item.casefold() for item in referenced_set}:
            continue
        citations.append({
            "source_number": index,
            "chunk_id": chunk_id,
            "document_id": source.document_id,
            "document_name": source.document_name,
            "page_number": source.page_number,
            "similarity": source.similarity,
            "snippet": source.content[:200] + ("..." if len(source.content) > 200 else ""),
        })

    if citation_issue:
        if re.search(r"[\u0600-\u06ff]", cleaned):
            note = "تنبيه: لم أتمكن من مطابقة كل استشهاد بالمقاطع المسترجعة؛ راجع الإجابة مع المادة المرفوعة."
        else:
            note = "Note: I could not match all citations to the retrieved passages; please check the answer against the uploaded material."
        cleaned = f"{cleaned}\n\n{note}" if cleaned else note

    return cleaned, citations, citation_issue


class EvidenceValidator:
    """Backward-compatible, local-only evidence check.

    Older callers passed a question and retrieved passages to an asynchronous
    validator. Keep that interface available without making a second model call:
    return SUPPORTED only when a question topic appears in the supplied text
    (including a locally derived acronym); otherwise fail closed.
    """

    def __init__(self, model_id: str | None = None):
        # Kept for source compatibility. Validation is deliberately provider-free.
        self.model_id = model_id

    async def validate(self, question: str, sources: list) -> str:
        question_terms = {
            token.casefold()
            for token in _WORD.findall(question or "")
            if token.casefold() not in _QUESTION_FILLER
        }
        if not question_terms or not sources:
            return "NOT_SUPPORTED"

        source_terms: set[str] = set()
        source_acronyms: set[str] = set()
        for source in sources:
            content = source if isinstance(source, str) else getattr(source, "content", "")
            words = _WORD.findall(content or "")
            source_terms.update(word.casefold() for word in words)

            # A simple local acronym lookup supports questions like “What does
            # NLP do?” when the passage spells out “natural language processing”.
            for width in range(2, min(6, len(words)) + 1):
                for start in range(len(words) - width + 1):
                    phrase = words[start:start + width]
                    if all(word[:1].isalpha() for word in phrase):
                        source_acronyms.add("".join(word[0] for word in phrase).casefold())

        if question_terms & (source_terms | source_acronyms):
            return "SUPPORTED"
        return "NOT_SUPPORTED"
