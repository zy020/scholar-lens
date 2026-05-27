from __future__ import annotations

import logging
import re
from pathlib import Path

from scholar_lens.parsers.models import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

_SLIDES_PDF_MAX_CHARS = 300
_FORMULA_PATTERN = re.compile(
    r"(?:softmax|LayerNorm|sqrt|frac|sum|prod|argmax|argmin|\\[a-zA-Z]+|[A-Za-z]\s*[\^_=]\s*[A-Za-z0-9({])",
    re.IGNORECASE,
)


def detect_pdf_subtype(pages: list[ParsedPage]) -> str:
    if not pages:
        return "research_paper"
    avg_chars = sum(p.char_count for p in pages) / len(pages)
    has_two_column = any(p.is_two_column for p in pages)
    has_abstract = any(p.has_abstract for p in pages)
    if avg_chars < _SLIDES_PDF_MAX_CHARS and not has_two_column:
        return "slides_pdf"
    if has_abstract or has_two_column:
        return "research_paper"
    return "research_paper"


def diagnose_text_quality(
    pages: list[ParsedPage],
    raw_text: str = "",
    sections: list[dict] | None = None,
) -> dict:
    sections = sections or []
    notes: list[str] = []

    if not pages:
        raw_len = len(raw_text or "")
        quality = "good" if raw_len >= 1000 else "unknown"
        if quality == "unknown":
            notes.append("未获得页面文本，无法可靠判断 PDF 文本质量。")
        return {
            "text_quality": quality,
            "ocr_needed": quality != "good",
            "page_text_coverage": 1.0 if quality == "good" else 0.0,
            "section_quality": "good" if len(sections) >= 3 else ("weak" if sections else "none"),
            "diagnostic_notes": notes,
        }

    total_pages = len(pages)
    char_counts = [p.char_count if p.char_count else len(p.text or "") for p in pages]
    avg_chars = sum(char_counts) / total_pages
    pages_with_text = sum(1 for count in char_counts if count >= 80)
    coverage = round(pages_with_text / total_pages, 2)
    raw_len = len(raw_text or "")

    if total_pages >= 3 and avg_chars < 80:
        quality = "image_based"
        notes.append("当前 PDF 疑似图片型课件，文本抽取不足，建议启用 OCR 或 Vision Model。")
    elif avg_chars < 220 or raw_len < 1000:
        quality = "weak"
        notes.append("当前 PDF 可抽取文本偏少，Study Brief 将使用保守模式，避免编造细节。")
    else:
        quality = "good"

    titled_sections = [s for s in sections if len(str(s.get("title", "")).strip()) >= 3]
    if len(titled_sections) >= 3:
        section_quality = "good"
    elif titled_sections:
        section_quality = "weak"
    else:
        section_quality = "none"
        if quality != "good":
            notes.append("未能可靠抽取章节结构。")

    return {
        "text_quality": quality,
        "ocr_needed": quality in {"weak", "image_based"},
        "page_text_coverage": coverage,
        "section_quality": section_quality,
        "diagnostic_notes": notes,
    }


class PDFParser:
    """Parses PDF documents with local text extractors."""

    def __init__(self, page_limit: int = 200) -> None:
        self._page_limit = page_limit

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        file_size_mb = pdf_path.stat().st_size / 1024 / 1024
        if self._page_limit and file_size_mb > 100:
            logger.warning(
                "Large file (%.0f MB), using PyMuPDF text extraction",
                file_size_mb,
            )
            return self._parse_with_pymupdf(pdf_path)

        try:
            return self._parse_with_pymupdf(pdf_path)
        except ImportError:
            logger.warning("PyMuPDF4LLM not installed, trying pdfplumber")
        except Exception as e:
            logger.warning(f"PyMuPDF4LLM parsing failed: {e}")

        return self._parse_with_pdfplumber(pdf_path)

    def _parse_with_pymupdf(self, pdf_path: Path) -> ParsedDocument:
        import fitz

        doc = fitz.open(str(pdf_path))
        pages = []
        images: list[dict] = []
        tables: list[dict] = []
        formulas: list[dict] = []
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))
            images.extend(_extract_page_images(page, i))
            tables.extend(_extract_page_tables(page, i))
            formulas.extend(_extract_formula_candidates(text, i))

        raw_text = "\n".join(p.text for p in pages)

        sections = self._extract_toc_sections(doc)
        doc.close()

        return ParsedDocument(
            source_path=str(pdf_path),
            parser_used="pymupdf",
            doc_subtype=detect_pdf_subtype(pages),
            pages=pages,
            sections=sections,
            raw_text=raw_text,
            formulas=formulas,
            tables=tables,
            images=images,
        )

    @staticmethod
    def _extract_toc_sections(doc) -> list[dict]:
        """Extract table-of-contents entries as section dicts with page numbers."""
        sections: list[dict] = []
        toc = doc.get_toc(simple=False)
        for i, item in enumerate(toc[:50]):
            if len(item) >= 3:
                sections.append({
                    "id": str(i + 1),
                    "title": str(item[1]),
                    "level": item[0],
                    "page_start": max(int(item[2]) - 1, 0),
                })
        return sections

    def _parse_with_pdfplumber(self, pdf_path: Path) -> ParsedDocument:
        import pdfplumber

        pages = []
        tables: list[dict] = []
        formulas: list[dict] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))
                formulas.extend(_extract_formula_candidates(text, i))
                try:
                    extracted_tables = page.extract_tables() or []
                except Exception:
                    extracted_tables = []
                for table_idx, table in enumerate(extracted_tables):
                    tables.append({
                        "page": i,
                        "table_index": table_idx,
                        "rows": len(table or []),
                        "source": "pdfplumber",
                    })

        raw_text = "\n".join(p.text for p in pages)
        return ParsedDocument(
            source_path=str(pdf_path),
            parser_used="pdfplumber",
            doc_subtype=detect_pdf_subtype(pages),
            pages=pages,
            tables=tables,
            formulas=formulas,
            raw_text=raw_text,
        )


def _extract_page_images(page, page_num: int) -> list[dict]:
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    images = []
    seen: set[tuple[float, float, float, float]] = set()
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = tuple(round(float(value), 2) for value in block.get("bbox", []))
        if len(bbox) != 4 or bbox in seen:
            continue
        seen.add(bbox)
        area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0.0)
        images.append({
            "page": page_num,
            "bbox": list(bbox),
            "area_ratio": round(area / page_area, 4),
            "source": "pymupdf",
        })
    if images:
        return images

    try:
        image_refs = page.get_images(full=True)
    except Exception:
        image_refs = []
    return [
        {
            "page": page_num,
            "image_index": idx,
            "source": "pymupdf",
        }
        for idx, _image in enumerate(image_refs)
    ]


def _extract_page_tables(page, page_num: int) -> list[dict]:
    if not hasattr(page, "find_tables"):
        return []
    try:
        found = page.find_tables()
    except Exception:
        return []
    tables = getattr(found, "tables", []) or []
    result = []
    for idx, table in enumerate(tables):
        bbox = getattr(table, "bbox", None)
        item = {
            "page": page_num,
            "table_index": idx,
            "source": "pymupdf",
        }
        if bbox:
            item["bbox"] = [round(float(value), 2) for value in bbox]
        result.append(item)
    return result


def _extract_formula_candidates(text: str, page_num: int) -> list[dict]:
    candidates = []
    for line in text.splitlines():
        normalized = line.strip()
        if len(normalized) < 4:
            continue
        if _FORMULA_PATTERN.search(normalized):
            candidates.append({
                "page": page_num,
                "text": normalized[:300],
                "source": "text_heuristic",
            })
    return candidates
