from __future__ import annotations

import logging
from pathlib import Path

from scholar_lens.parsers.models import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

_SLIDES_PDF_MAX_CHARS = 300


def detect_pdf_subtype(pages: list[ParsedPage]) -> str:
    if not pages:
        return "general_document"
    avg_chars = sum(p.char_count for p in pages) / len(pages)
    has_two_column = any(p.is_two_column for p in pages)
    has_abstract = any(p.has_abstract for p in pages)
    if avg_chars < _SLIDES_PDF_MAX_CHARS and not has_two_column:
        return "slides_pdf"
    if has_abstract or has_two_column:
        return "research_paper"
    return "general_document"


class PDFParser:
    """Parses PDF documents using Docling with fallback chain."""

    def __init__(self, use_docling: bool = True) -> None:
        self._use_docling = use_docling

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self._use_docling:
            try:
                return self._parse_with_docling(pdf_path)
            except ImportError:
                logger.warning("docling not installed, trying fallback parsers")
            except Exception as e:
                logger.warning(f"Docling parsing failed: {e}, trying fallback")

        try:
            return self._parse_with_pymupdf(pdf_path)
        except ImportError:
            logger.warning("PyMuPDF4LLM not installed, trying pdfplumber")
        except Exception as e:
            logger.warning(f"PyMuPDF4LLM parsing failed: {e}")

        return self._parse_with_pdfplumber(pdf_path)

    def _parse_with_docling(self, pdf_path: Path) -> ParsedDocument:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        pages = []
        for i, page in enumerate(doc.pages.values()):
            text = page.text if hasattr(page, "text") else ""
            pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))

        sections = []
        if hasattr(doc, "texts"):
            for item in doc.texts:
                if hasattr(item, "heading") and item.heading:
                    sections.append({
                        "id": item.heading,
                        "title": item.heading,
                        "level": getattr(item, "level", 1),
                        "text": item.text if hasattr(item, "text") else "",
                    })

        raw_text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else "\n".join(p.text for p in pages)
        return ParsedDocument(
            source_path=str(pdf_path),
            doc_subtype=detect_pdf_subtype(pages),
            pages=pages,
            sections=sections,
            raw_text=raw_text,
        )

    def _parse_with_pymupdf(self, pdf_path: Path) -> ParsedDocument:
        import fitz

        doc = fitz.open(str(pdf_path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))

        raw_text = "\n".join(p.text for p in pages)
        return ParsedDocument(
            source_path=str(pdf_path),
            doc_subtype=detect_pdf_subtype(pages),
            pages=pages,
            raw_text=raw_text,
        )

    def _parse_with_pdfplumber(self, pdf_path: Path) -> ParsedDocument:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))

        raw_text = "\n".join(p.text for p in pages)
        return ParsedDocument(
            source_path=str(pdf_path),
            doc_subtype=detect_pdf_subtype(pages),
            pages=pages,
            raw_text=raw_text,
        )
