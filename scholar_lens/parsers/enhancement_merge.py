from __future__ import annotations

from pydantic import BaseModel

from scholar_lens.parsers.models import ParsedDocument


class EnhancementFragment(BaseModel):
    page: int
    source: str
    text: str
    quality: str
    reason: str = ""


def fragments_from_ocr_payload(payload: dict) -> list[EnhancementFragment]:
    fragments: list[EnhancementFragment] = []
    for page in payload.get("pages", []):
        fragments.append(EnhancementFragment(
            page=int(page.get("page", 0)),
            source="ocr",
            text=str(page.get("text", "")),
            quality=str(page.get("ocr_quality", page.get("quality", "failed"))),
            reason=str(page.get("reason", "")),
        ))
    return fragments


def fragments_from_vision_payload(payload: dict) -> list[EnhancementFragment]:
    fragments: list[EnhancementFragment] = []
    for page in payload.get("pages", []):
        fragments.append(EnhancementFragment(
            page=int(page.get("page", 0)),
            source="vision",
            text=str(page.get("text", "")),
            quality=str(page.get("vision_quality", page.get("quality", "failed"))),
            reason=str(page.get("reason", "")),
        ))
    return fragments


def merge_enhancements(
    doc: ParsedDocument,
    fragments: list[EnhancementFragment],
) -> ParsedDocument:
    merged = doc.model_copy(deep=True)
    fragments_by_page = _usable_fragments_by_page(fragments)
    for page in merged.pages:
        page_fragments = fragments_by_page.get(page.page_num, [])
        if not page_fragments:
            continue
        text = page.text or ""
        for fragment in page_fragments:
            text = _merge_page_text(text, fragment)
            page.content_source = _merged_content_source(page.content_source, fragment.source, page.enhanced)
            page.enhanced = True
        page.text = text
        page.char_count = len(text)

    merged.raw_text = "\n\n".join(page.text.strip() for page in merged.pages if page.text.strip())
    if merged.doc_subtype in {"slides_pdf", "courseware_pptx"}:
        merged.sections = _courseware_sections(merged)
    return merged


def _usable_fragments_by_page(fragments: list[EnhancementFragment]) -> dict[int, list[EnhancementFragment]]:
    grouped: dict[int, list[EnhancementFragment]] = {}
    for fragment in fragments:
        if fragment.quality == "failed":
            continue
        if not fragment.text.strip():
            continue
        grouped.setdefault(fragment.page, []).append(fragment)
    return grouped


def _merge_page_text(existing: str, fragment: EnhancementFragment) -> str:
    text = fragment.text.strip()
    if fragment.source == "ocr" and (not existing.strip() or len(existing.strip()) < 20):
        return text
    if not existing.strip():
        return text
    marker = f"[{fragment.source.upper()}]"
    return f"{existing.rstrip()}\n\n{marker}\n{text}"


def _courseware_sections(doc: ParsedDocument) -> list[dict]:
    sections = []
    for page in doc.pages:
        text = page.text.strip()
        if not text:
            continue
        sections.append({
            "id": f"slide_{page.page_num}",
            "title": f"Slide {page.page_num + 1}",
            "level": 1,
            "text": text,
            "page_start": page.page_num,
            "page_end": page.page_num,
            "content_source": page.content_source,
            "enhanced": page.enhanced,
        })
    return sections


def _merged_content_source(existing_source: str, fragment_source: str, already_enhanced: bool) -> str:
    source = fragment_source or "parser"
    if not already_enhanced or existing_source in ("", "parser"):
        return source
    if existing_source == source:
        return existing_source
    return "mixed"


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""
