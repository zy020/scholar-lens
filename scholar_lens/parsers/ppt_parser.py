from __future__ import annotations

import logging
from pathlib import Path

from scholar_lens.parsers.models import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)


class PPTParser:
    """Parses PPTX files using python-pptx, with Docling as optional fallback."""

    def parse(self, ppt_path: str | Path) -> ParsedDocument:
        ppt_path = Path(ppt_path)
        if not ppt_path.exists():
            raise FileNotFoundError(f"PPTX not found: {ppt_path}")

        try:
            return self._parse_with_pptx(ppt_path)
        except ImportError:
            logger.warning("python-pptx not installed, trying docling fallback")
            return self._parse_with_docling(ppt_path)

    def _parse_with_pptx(self, ppt_path: Path) -> ParsedDocument:
        from pptx import Presentation

        prs = Presentation(str(ppt_path))
        pages = []
        sections = []

        for i, slide in enumerate(prs.slides):
            texts = []
            title = ""

            for shape in slide.shapes:
                if shape.has_text_frame:
                    shape_text = shape.text_frame.text.strip()
                    if shape_text:
                        texts.append(shape_text)
                        if not title and shape.shape_type == 14:
                            title = shape_text

            notes = ""
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                notes = notes_slide.notes_text_frame.text.strip()

            slide_text = "\n".join(texts)
            if notes:
                slide_text += f"\n\n[Speaker Notes]\n{notes}"

            pages.append(ParsedPage(page_num=i, text=slide_text, char_count=len(slide_text)))

            if title:
                sections.append({
                    "id": f"slide_{i}",
                    "title": title,
                    "level": 1,
                    "text": slide_text,
                    "page_start": i,
                    "page_end": i,
                })

        raw_text = "\n\n".join(p.text for p in pages)
        return ParsedDocument(
            source_path=str(ppt_path),
            doc_subtype="courseware_pptx",
            pages=pages,
            sections=sections,
            raw_text=raw_text,
        )

    def _parse_with_docling(self, ppt_path: Path) -> ParsedDocument:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(ppt_path))
        doc = result.document

        pages = []
        for i, page in enumerate(doc.pages.values()):
            text = page.text if hasattr(page, "text") else ""
            pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))

        raw_text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else "\n".join(p.text for p in pages)

        return ParsedDocument(
            source_path=str(ppt_path),
            doc_subtype="courseware_pptx",
            pages=pages,
            raw_text=raw_text,
        )
