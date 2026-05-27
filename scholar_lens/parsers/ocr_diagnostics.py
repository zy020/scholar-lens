from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from scholar_lens.parsers.models import ParsedDocument

LOW_TEXT_CHARS = 20
NEAR_EMPTY_CHARS = 4
VISUAL_DENSITY_THRESHOLD = 0.18
IMAGE_BASED_MIN_PAGES = 3
IMAGE_BASED_LOW_TEXT_RATIO = 0.6
IMAGE_BASED_NEAR_EMPTY_RATIO = 0.5


class OCRRecommendation(BaseModel):
    pages: list[int] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)


class OCRQualityResult(BaseModel):
    ocr_quality: str
    vision_recommended: bool = False
    reason: str = ""
    usable_text: str = ""


def recommend_ocr_pages(
    doc: ParsedDocument,
    visual_density_by_page: Mapping[int, float] | None = None,
) -> OCRRecommendation:
    visual_density_by_page = visual_density_by_page or {}
    visual_pages = _visual_pages(doc)
    low_text_pages = [
        page.page_num
        for page in doc.pages
        if _page_text_len(page) < LOW_TEXT_CHARS
    ]

    if _is_document_image_based(doc, low_text_pages):
        return _recommend(low_text_pages, "document_image_based")

    reasons: dict[str, str] = {}
    for page in doc.pages:
        page_num = page.page_num
        if _page_text_len(page) >= LOW_TEXT_CHARS:
            continue
        if page_num in visual_pages:
            reasons[str(page_num)] = "text_low_parser_visuals"
            continue
        if visual_density_by_page.get(page_num, 0.0) >= VISUAL_DENSITY_THRESHOLD:
            reasons[str(page_num)] = "text_low_visual_high"

    pages = sorted(int(page) for page in reasons)
    return OCRRecommendation(pages=pages, reasons={str(page): reasons[str(page)] for page in pages})


def evaluate_ocr_result(
    ocr_text: str,
    visual_density: float = 0.0,
    page_kind: str = "text",
) -> OCRQualityResult:
    usable_text = " ".join((ocr_text or "").split())
    if len(usable_text) < 8 and visual_density >= VISUAL_DENSITY_THRESHOLD:
        return OCRQualityResult(
            ocr_quality="weak",
            vision_recommended=True,
            reason="ocr_too_short_visual_high",
            usable_text=usable_text,
        )

    if len(usable_text) < 8:
        return OCRQualityResult(
            ocr_quality="failed",
            vision_recommended=True,
            reason="ocr_too_short",
            usable_text=usable_text,
        )

    if _symbol_ratio(usable_text) >= 0.55:
        return OCRQualityResult(
            ocr_quality="failed",
            vision_recommended=True,
            reason="garbled_text",
            usable_text=usable_text,
        )

    if page_kind in {"diagram", "chart", "formula", "flow"}:
        return OCRQualityResult(
            ocr_quality="weak",
            vision_recommended=True,
            reason="diagram_like",
            usable_text=usable_text,
        )

    if len(usable_text) < 40 and visual_density >= VISUAL_DENSITY_THRESHOLD:
        return OCRQualityResult(
            ocr_quality="weak",
            vision_recommended=True,
            reason="ocr_too_short_visual_high",
            usable_text=usable_text,
        )

    return OCRQualityResult(
        ocr_quality="good",
        vision_recommended=False,
        reason="ocr_text_usable",
        usable_text=usable_text,
    )


def _symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    symbol_chars = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbol_chars / len(text)


def _page_text_len(page) -> int:
    return page.char_count if page.char_count else len(page.text or "")


def _is_document_image_based(doc: ParsedDocument, low_text_pages: list[int]) -> bool:
    total_pages = len(doc.pages)
    near_empty_pages = [
        page
        for page in doc.pages
        if _page_text_len(page) <= NEAR_EMPTY_CHARS
    ]
    return (
        doc.doc_subtype in {"slides_pdf", "courseware_pptx"}
        and total_pages >= IMAGE_BASED_MIN_PAGES
        and len(low_text_pages) / total_pages >= IMAGE_BASED_LOW_TEXT_RATIO
        and len(near_empty_pages) / total_pages >= IMAGE_BASED_NEAR_EMPTY_RATIO
    )


def _recommend(pages: list[int], reason: str) -> OCRRecommendation:
    sorted_pages = sorted(pages)
    return OCRRecommendation(
        pages=sorted_pages,
        reasons={str(page): reason for page in sorted_pages},
    )


def _visual_pages(doc: ParsedDocument) -> set[int]:
    pages: set[int] = set()
    for item in [*doc.images, *doc.tables, *doc.formulas]:
        page = _extract_page_num(item)
        if page is not None:
            pages.add(page)
    return pages


def _extract_page_num(item: dict) -> int | None:
    for key in ("page", "page_num", "page_start"):
        value = item.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None
