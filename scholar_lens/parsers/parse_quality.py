from __future__ import annotations

from pydantic import BaseModel, Field

from scholar_lens.parsers.models import ParsedDocument, ParsedPage
from scholar_lens.parsers.ocr_diagnostics import OCRRecommendation


class ParseUnitQuality(BaseModel):
    unit_id: str
    unit_type: str = "page"
    page_start: int
    page_end: int
    text_score: float = 0.0
    structure_score: float = 0.0
    visual_score: float = 0.0
    ocr_need_score: float = 0.0
    overall_score: float = 0.0
    quality: str = "failed"
    recommended_action: str = "keep"
    reasons: list[str] = Field(default_factory=list)
    text_preview: str = ""


def assess_parse_unit_quality(doc: ParsedDocument) -> list[ParseUnitQuality]:
    visual_pages = _visual_pages(doc)
    formula_pages = _formula_pages(doc)
    table_pages = _table_pages(doc)
    image_pages = _image_pages(doc)
    qualities = []
    for page in doc.pages:
        qualities.append(_assess_page(doc, page, visual_pages, formula_pages, table_pages, image_pages))
    return qualities


def recommend_ocr_from_quality(qualities: list[ParseUnitQuality]) -> OCRRecommendation:
    pages = []
    reasons = {}
    for quality in qualities:
        if quality.recommended_action != "ocr":
            continue
        page = quality.page_start
        pages.append(page)
        if "visual_high" in quality.reasons:
            reasons[str(page)] = "text_low_parser_visuals"
        else:
            reasons[str(page)] = "document_image_based"
    if not pages:
        for quality in _document_low_text_slide_candidates(qualities):
            page = quality.page_start
            pages.append(page)
            reasons[str(page)] = "document_low_text_slides"
    return OCRRecommendation(pages=pages, reasons=reasons)


def _document_low_text_slide_candidates(qualities: list[ParseUnitQuality]) -> list[ParseUnitQuality]:
    slide_qualities = [quality for quality in qualities if quality.unit_type == "slide"]
    if len(slide_qualities) < 3:
        return []
    low_text = [quality for quality in slide_qualities if quality.text_score < 0.35]
    if len(low_text) / len(slide_qualities) < 0.6:
        return []
    return [
        quality
        for quality in low_text
        if _has_content_trace(quality.text_preview)
    ]


def _has_content_trace(text: str) -> bool:
    normalized = " ".join((text or "").split())
    if len(normalized) < 12:
        return False
    return normalized.lower() not in {"agenda", "summary", "outline", "overview"}


def _assess_page(
    doc: ParsedDocument,
    page: ParsedPage,
    visual_pages: set[int],
    formula_pages: set[int] | None = None,
    table_pages: set[int] | None = None,
    image_pages: set[int] | None = None,
) -> ParseUnitQuality:
    formula_pages = formula_pages or set()
    table_pages = table_pages or set()
    image_pages = image_pages or set()
    text = page.text or ""
    text_len = page.char_count if page.char_count else len(text)
    expected_chars = _expected_chars(doc.doc_subtype)
    text_score = _clamp(text_len / expected_chars)
    text_score = _clamp(text_score - _symbol_ratio(text) * 0.5 - _repetition_ratio(text) * 0.3)

    has_title = bool(_first_nonempty_line(text))
    line_count = len([line for line in text.splitlines() if line.strip()])
    structure_score = 0.3
    if has_title:
        structure_score += 0.25
    if line_count >= 2:
        structure_score += 0.2
    if doc.doc_subtype in {"slides_pdf", "courseware_pptx"}:
        structure_score += 0.2
    structure_score = _clamp(structure_score)

    visual_score = 0.75 if page.page_num in visual_pages else 0.0
    reasons = []
    if text_score < 0.3:
        reasons.append("text_low")
    if visual_score >= 0.5:
        reasons.append("visual_high")
    if page.page_num in formula_pages and text_len < 20:
        reasons.append("formula_signal")
    if page.page_num in table_pages and text_len < 40:
        reasons.append("complex_table_signal")
    if page.page_num in image_pages and text_len < 20:
        reasons.append("diagram_signal")

    ocr_need_score = 0.0
    if text_score < 0.3:
        ocr_need_score += 0.45
    if visual_score > 0.4:
        ocr_need_score += 0.35
    if structure_score < 0.3:
        ocr_need_score += 0.15

    if text_score < 0.3 and visual_score < 0.2 and has_title:
        ocr_need_score = max(0.0, ocr_need_score - 0.35)
        reasons.append("sparse_but_structured")
    if page.page_num in formula_pages and text_len >= 20:
        ocr_need_score = max(0.0, ocr_need_score - 0.45)
        reasons.append("formula_text_available")

    overall_score = _clamp(0.65 * text_score + 0.35 * structure_score)
    if visual_score > 0.5 and text_score < 0.4:
        overall_score = _clamp(overall_score - 0.25)

    quality = _quality_label(overall_score)
    recommended_action = "keep"
    formula_text_available = "formula_text_available" in reasons
    if formula_text_available:
        recommended_action = "keep"
    elif ocr_need_score >= 0.5 and visual_score > 0.4:
        recommended_action = "ocr"
    elif text_score < 0.2 and visual_score < 0.2:
        recommended_action = "keep"
    elif overall_score < 0.35 and visual_score > 0.5:
        recommended_action = "ocr"

    return ParseUnitQuality(
        unit_id=f"page_{page.page_num}",
        unit_type="slide" if doc.doc_subtype in {"slides_pdf", "courseware_pptx"} else "page",
        page_start=page.page_num,
        page_end=page.page_num,
        text_score=round(text_score, 3),
        structure_score=round(structure_score, 3),
        visual_score=round(visual_score, 3),
        ocr_need_score=round(_clamp(ocr_need_score), 3),
        overall_score=round(overall_score, 3),
        quality=quality,
        recommended_action=recommended_action,
        reasons=reasons,
        text_preview=" ".join(text.split())[:180],
    )


def _expected_chars(doc_subtype: str) -> int:
    if doc_subtype == "research_paper":
        return 800
    if doc_subtype in {"slides_pdf", "courseware_pptx"}:
        return 120
    return 400


def _visual_pages(doc: ParsedDocument) -> set[int]:
    pages: set[int] = set()
    for item in [*doc.images, *doc.tables, *doc.formulas]:
        page = item.get("page", item.get("page_num", item.get("page_start")))
        if isinstance(page, int):
            pages.add(page)
        elif isinstance(page, str) and page.isdigit():
            pages.add(int(page))
    return pages


def _formula_pages(doc: ParsedDocument) -> set[int]:
    return _pages_from_items(doc.formulas)


def _table_pages(doc: ParsedDocument) -> set[int]:
    return _pages_from_items(doc.tables)


def _image_pages(doc: ParsedDocument) -> set[int]:
    return _pages_from_items(doc.images)


def _pages_from_items(items: list[dict]) -> set[int]:
    pages: set[int] = set()
    for item in items:
        page = item.get("page", item.get("page_num", item.get("page_start")))
        if isinstance(page, int):
            pages.add(page)
        elif isinstance(page, str) and page.isdigit():
            pages.add(int(page))
    return pages


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbol_count / len(text)


def _repetition_ratio(text: str) -> float:
    words = [word for word in text.lower().split() if word.strip()]
    if len(words) < 12:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    return max(0.0, 0.5 - unique_ratio)


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _quality_label(score: float) -> str:
    if score >= 0.7:
        return "good"
    if score >= 0.35:
        return "weak"
    return "failed"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
