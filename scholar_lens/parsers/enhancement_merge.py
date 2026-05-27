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
        candidate = _select_page_candidate(page.text or "", page_fragments)
        if candidate.source == "parser":
            continue
        text = candidate.text
        page.content_source = candidate.source
        page.enhanced = True
        page.text = text
        page.char_count = len(text)

    merged.raw_text = "\n\n".join(page.text.strip() for page in merged.pages if page.text.strip())
    if merged.doc_subtype in {"slides_pdf", "courseware_pptx"}:
        merged.sections = _courseware_sections(merged)
    return merged


class _PageCandidate(BaseModel):
    source: str
    text: str
    score: float


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


def _select_page_candidate(existing: str, fragments: list[EnhancementFragment]) -> _PageCandidate:
    existing_text = existing.strip()
    candidates = [_PageCandidate(source="parser", text=existing_text, score=_candidate_score(existing_text, "parser", "good"))]
    for fragment in fragments:
        fragment_text = fragment.text.strip()
        candidates.append(_PageCandidate(
            source=fragment.source,
            text=fragment_text,
            score=_candidate_score(fragment_text, fragment.source, fragment.quality),
        ))
        merged_text = _merge_page_text(existing_text, fragment)
        merge_bonus = 0.16 if existing_text and fragment.quality == "good" else 0.0
        candidates.append(_PageCandidate(
            source=_merged_content_source("parser", fragment.source, False),
            text=merged_text,
            score=_candidate_score(merged_text, fragment.source, fragment.quality) + merge_bonus - 0.08,
        ))
    return max(candidates, key=lambda item: item.score)


def _candidate_score(text: str, source: str, quality: str) -> float:
    stripped = " ".join((text or "").split())
    if not stripped:
        return 0.0
    length_score = min(1.0, len(stripped) / 180)
    symbol_penalty = _symbol_ratio(stripped) * 0.5
    fragmentation_penalty = _fragmentation(stripped) * 0.4
    quality_bonus = {"good": 0.25, "weak": 0.02, "failed": -0.4}.get(quality, 0.0)
    source_bonus = {"vision": 0.12, "ocr": 0.04, "parser": 0.08}.get(source, 0.0)
    return max(0.0, min(1.0, length_score + quality_bonus + source_bonus - symbol_penalty - fragmentation_penalty))


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbols / len(text)


def _fragmentation(text: str) -> float:
    tokens = [token for token in text.split() if token.strip()]
    if not tokens:
        return 1.0
    short = sum(1 for token in tokens if len(token.strip(".,;:()[]{}")) <= 1)
    noisy = sum(1 for token in tokens if any(ch in token for ch in "?|�"))
    return min(1.0, (short + noisy) / max(1, len(tokens)))


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
