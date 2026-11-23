"""
Document text extraction.

Supports: PDF (PyMuPDF), TXT, Markdown, CSV, XLSX.
Each extractor returns a list of Page objects (text + page number).
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """One logical page/section of extracted text."""
    text: str
    page_number: int
    metadata: dict = field(default_factory=dict)


class DocumentExtractor:
    """Dispatches extraction based on file type."""

    SUPPORTED_TYPES = {"pdf", "txt", "md", "csv", "xlsx"}

    @classmethod
    def extract(cls, file_path: str, file_type: str) -> list[ExtractedPage]:
        """
        Extract text from a file.

        Args:
            file_path: Absolute path to the stored file.
            file_type: Extension without dot (pdf, txt, md, csv, xlsx).

        Returns:
            List of ExtractedPage with text and page numbers.

        Raises:
            ValueError: If the file type is not supported.
        """
        file_type = file_type.lower().strip(".")
        if file_type not in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type: {file_type}. "
                f"Supported: {', '.join(sorted(cls.SUPPORTED_TYPES))}"
            )

        handler = {
            "pdf": cls._extract_pdf,
            "txt": cls._extract_text,
            "md": cls._extract_text,
            "csv": cls._extract_csv,
            "xlsx": cls._extract_xlsx,
        }[file_type]

        pages = handler(file_path)
        logger.info("Extracted %d pages from %s (%s)", len(pages), file_path, file_type)
        return pages

    @staticmethod
    def _extract_pdf(file_path: str) -> list[ExtractedPage]:
        """Extract text from PDF using PyMuPDF (preserves layout)."""
        import fitz  # PyMuPDF

        pages = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append(ExtractedPage(
                        text=text.strip(),
                        page_number=page_num,
                        metadata={"total_pages": len(doc)},
                    ))
        return pages

    @staticmethod
    def _extract_text(file_path: str) -> list[ExtractedPage]:
        """Extract text from plain text or markdown files."""
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []
        return [ExtractedPage(text=text.strip(), page_number=1)]

    @staticmethod
    def _extract_csv(file_path: str) -> list[ExtractedPage]:
        """
        Extract CSV — one page per row.

        Each row becomes a structured text with column headers,
        making it easy to retrieve individual records.
        """
        pages = []
        path = Path(file_path)
        with path.open(encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # row 2 = first data row
                parts = [f"{col}: {val}" for col, val in row.items() if val]
                if parts:
                    pages.append(ExtractedPage(
                        text=" | ".join(parts),
                        page_number=row_num,
                        metadata={"source_type": "csv_row"},
                    ))
        return pages

    @staticmethod
    def _extract_xlsx(file_path: str) -> list[ExtractedPage]:
        """
        Extract XLSX — one page per row, across all sheets.

        Row numbers match Excel's numbering (header = row 1).
        """
        from openpyxl import load_workbook

        pages = []
        wb = load_workbook(file_path, read_only=True, data_only=True)

        for sheet in wb.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if len(rows) < 2:
                continue

            # First row = headers
            headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]

            for row_num, row in enumerate(rows[1:], start=2):
                parts = []
                for col_name, val in zip(headers, row):
                    if val is not None:
                        parts.append(f"{col_name}: {val}")
                if parts:
                    pages.append(ExtractedPage(
                        text=" | ".join(parts),
                        page_number=row_num,
                        metadata={
                            "source_type": "xlsx_row",
                            "sheet_name": sheet.title,
                        },
                    ))

        wb.close()
        return pages
