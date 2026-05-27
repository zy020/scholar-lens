from __future__ import annotations

from pydantic import BaseModel, Field

from scholar_lens.parsers.parse_quality import ParseUnitQuality


class EnhancementDecision(BaseModel):
    page: int
    action: str = "use_original"
    reason: str = "quality_good"
    reasons: list[str] = Field(default_factory=list)
    parser_score: float = 0.0
    visual_score: float = 0.0
    ocr_readability: float = 0.0
    ocr_gain: float = 0.0
    ocr_fragmentation: float = 0.0


def build_enhancement_decisions(
    qualities: list[ParseUnitQuality | dict],
    *,
    ocr_payload: dict | None = None,
    vision_enabled: bool = False,
) -> list[EnhancementDecision]:
    ocr_by_page = _ocr_pages(ocr_payload or {})
    decisions: list[EnhancementDecision] = []
    for raw_quality in qualities:
        quality = _coerce_quality(raw_quality)
        page = quality.page_start
        ocr_page = ocr_by_page.get(page)
        metrics = _ocr_metrics(quality, ocr_page)
        action, reason = _decide_action(quality, metrics, vision_enabled=vision_enabled, has_ocr_probe=ocr_page is not None)
        decisions.append(EnhancementDecision(
            page=page,
            action=action,
            reason=reason,
            reasons=list(quality.reasons),
            parser_score=quality.overall_score,
            visual_score=quality.visual_score,
            ocr_readability=round(metrics["readability"], 3),
            ocr_gain=round(metrics["gain"], 3),
            ocr_fragmentation=round(metrics["fragmentation"], 3),
        ))
    return decisions


def ocr_pages_from_decisions(decisions: list[EnhancementDecision]) -> list[int]:
    return [item.page for item in decisions if item.action in {"apply_ocr", "apply_ocr_then_maybe_vision"}]


def vision_pages_from_decisions(decisions: list[EnhancementDecision]) -> list[int]:
    return [item.page for item in decisions if item.action == "apply_vision"]


def _decide_action(
    quality: ParseUnitQuality,
    metrics: dict[str, float],
    *,
    vision_enabled: bool,
    has_ocr_probe: bool,
) -> tuple[str, str]:
    reasons = set(quality.reasons)
    if quality.recommended_action == "keep" or quality.overall_score >= 0.7:
        return "use_original", "quality_good"
    if "sparse_but_structured" in reasons:
        return "use_original", "sparse_but_structured"

    semantic_reasons = {"diagram_signal", "chart_signal", "formula_signal", "complex_table_signal"}
    needs_visual_semantics = bool(reasons & semantic_reasons)
    if vision_enabled and needs_visual_semantics and (not has_ocr_probe or metrics["readability"] < 0.65 or metrics["fragmentation"] > 0.35):
        return "apply_vision", "visual_semantics_need_vision"

    if has_ocr_probe:
        if metrics["readability"] >= 0.65 and metrics["gain"] >= 0.2:
            return "apply_ocr", "ocr_readable_gain"
        if vision_enabled and quality.visual_score >= 0.5:
            return "apply_vision", "ocr_probe_failed_visual_high"
        return "apply_ocr", "ocr_best_available"

    if vision_enabled and quality.visual_score >= 0.5:
        return "apply_ocr_then_maybe_vision", "ocr_first_for_visual_text"
    if quality.recommended_action == "ocr":
        return "apply_ocr", "parser_recommends_ocr"
    return "use_original", "quality_not_low"


def _ocr_pages(payload: dict) -> dict[int, dict]:
    pages: dict[int, dict] = {}
    for item in payload.get("pages", []) if isinstance(payload, dict) else []:
        page = item.get("page")
        if isinstance(page, int):
            pages[page] = item
        elif isinstance(page, str) and page.isdigit():
            pages[int(page)] = item
    return pages


def _ocr_metrics(quality: ParseUnitQuality, ocr_page: dict | None) -> dict[str, float]:
    if not ocr_page:
        return {"readability": 0.0, "gain": 0.0, "fragmentation": 0.0}
    text = str(ocr_page.get("text", "") or "")
    quality_label = str(ocr_page.get("ocr_quality", ocr_page.get("quality", "")))
    readability = _text_readability(text)
    if quality_label == "good":
        readability = max(readability, 0.75)
    elif quality_label == "failed":
        readability = min(readability, 0.2)
    parser_chars = max(1, int((quality.text_score or 0.0) * _expected_chars(quality.unit_type)))
    gain = _clamp((len(text.strip()) - parser_chars) / max(80, parser_chars))
    return {
        "readability": readability,
        "gain": gain,
        "fragmentation": _fragmentation(text),
    }


def _text_readability(text: str) -> float:
    stripped = " ".join(text.split())
    if not stripped:
        return 0.0
    length_score = _clamp(len(stripped) / 160)
    symbol_penalty = _symbol_ratio(stripped) * 0.7
    fragmentation_penalty = _fragmentation(stripped) * 0.5
    return _clamp(length_score - symbol_penalty - fragmentation_penalty)


def _fragmentation(text: str) -> float:
    tokens = [token for token in text.split() if token.strip()]
    if not tokens:
        return 1.0
    short = sum(1 for token in tokens if len(token.strip(".,;:()[]{}")) <= 1)
    noisy = sum(1 for token in tokens if any(ch in token for ch in "?|�"))
    return _clamp((short + noisy) / max(1, len(tokens)))


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbols / len(text)


def _expected_chars(unit_type: str) -> int:
    return 120 if unit_type == "slide" else 800


def _coerce_quality(raw_quality: ParseUnitQuality | dict) -> ParseUnitQuality:
    if isinstance(raw_quality, ParseUnitQuality):
        return raw_quality
    return ParseUnitQuality(**raw_quality)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
