"""
Text chunking with undersized-chunk merging.

Uses LangChain's RecursiveCharacterTextSplitter to break extracted
pages into retrieval-sized pieces, then merges any debris chunks
(running headers, page numbers) into their neighbours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from educore.config import settings
from educore.rag.extractor import ExtractedPage

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """One retrieval unit."""
    content: str
    page_number: int | None
    chunk_order: int = 0
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Split extracted pages into retrieval-sized chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_chars: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_chunk_chars = min_chunk_chars or settings.MIN_CHUNK_CHARS

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, pages: list[ExtractedPage]) -> list[TextChunk]:
        """
        Split pages → chunks → merge undersized → assign order.

        Args:
            pages: Extracted pages from a document.

        Returns:
            Ordered list of TextChunk ready for embedding.
        """
        if not pages:
            return []

        raw_chunks: list[TextChunk] = []

        for page in pages:
            text = self._sanitize(page.text)
            if not text:
                continue

            splits = self.splitter.split_text(text)

            for split in splits:
                raw_chunks.append(TextChunk(
                    content=split,
                    page_number=page.page_number,
                    metadata=page.metadata.copy(),
                ))

        # Merge undersized chunks into neighbours
        merged = self._merge_undersized(raw_chunks)

        # Assign sequential order
        for i, chunk in enumerate(merged):
            chunk.chunk_order = i

        logger.info(
            "Chunked %d pages → %d raw → %d final chunks",
            len(pages), len(raw_chunks), len(merged),
        )
        return merged

    def _merge_undersized(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """
        Merge debris chunks (too short to be useful) into neighbours.

        A chunk shorter than MIN_CHUNK_CHARS is appended to the
        previous chunk if they share the same page, or prepended
        to the next one otherwise.
        """
        if not chunks:
            return []

        merged: list[TextChunk] = []

        for chunk in chunks:
            if len(chunk.content) < self.min_chunk_chars and merged:
                # Append to previous chunk
                prev = merged[-1]
                prev.content = prev.content.rstrip() + "\n" + chunk.content
            else:
                merged.append(chunk)

        # If the last chunk is still too small, merge into the one before it
        if len(merged) > 1 and len(merged[-1].content) < self.min_chunk_chars:
            tail = merged.pop()
            merged[-1].content = merged[-1].content.rstrip() + "\n" + tail.content

        return merged

    @staticmethod
    def _sanitize(text: str) -> str:
        """
        Clean text before splitting.

        - NFKC normalise (maps Arabic presentation forms to standard)
        - Strip NUL bytes (PostgreSQL rejects them)
        - Collapse excessive whitespace
        """
        import unicodedata

        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\x00", "")

        # Collapse runs of blank lines to at most two
        lines = text.splitlines()
        cleaned: list[str] = []
        blank_count = 0
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append("")
            else:
                blank_count = 0
                cleaned.append(line)

        return "\n".join(cleaned).strip()
