from scholar_lens.parsers.enhancement_decision import build_enhancement_decisions
from scholar_lens.parsers.parse_quality import ParseUnitQuality


def quality(page, *, text_score=0.2, visual_score=0.8, overall_score=0.25, reasons=None):
    return ParseUnitQuality(
        unit_id=f"page_{page}",
        unit_type="slide",
        page_start=page,
        page_end=page,
        text_score=text_score,
        structure_score=0.4,
        visual_score=visual_score,
        ocr_need_score=0.8,
        overall_score=overall_score,
        quality="failed" if overall_score < 0.35 else "weak",
        recommended_action="ocr",
        reasons=reasons or ["text_low", "visual_high"],
        text_preview="",
    )


def test_decision_keeps_sparse_but_structured_page():
    decisions = build_enhancement_decisions([
        quality(0, text_score=0.1, visual_score=0.0, reasons=["text_low", "sparse_but_structured"]),
    ])

    assert decisions[0].action == "use_original"
    assert decisions[0].reason == "sparse_but_structured"


def test_decision_uses_ocr_when_probe_is_readable_and_has_gain():
    decisions = build_enhancement_decisions(
        [quality(2, text_score=0.1, visual_score=0.7)],
        ocr_payload={
            "pages": [
                {"page": 2, "text": "Self-attention maps queries to keys and values. " * 3, "ocr_quality": "good"},
            ]
        },
    )

    assert decisions[0].action == "apply_ocr"
    assert decisions[0].reason == "ocr_readable_gain"
    assert decisions[0].ocr_readability >= 0.7


def test_decision_escalates_to_vision_when_ocr_probe_is_fragmented_on_diagram_page():
    decisions = build_enhancement_decisions(
        [quality(3, reasons=["text_low", "visual_high", "diagram_signal"])],
        ocr_payload={
            "pages": [
                {"page": 3, "text": "Q K V -> -> ? ? | |", "ocr_quality": "weak"},
            ]
        },
        vision_enabled=True,
    )

    assert decisions[0].action == "apply_vision"
    assert decisions[0].reason == "visual_semantics_need_vision"
    assert "diagram_signal" in decisions[0].reasons


def test_decision_runs_ocr_then_maybe_vision_without_probe_for_visual_text_page():
    decisions = build_enhancement_decisions([
        quality(4, reasons=["text_low", "visual_high"]),
    ], vision_enabled=True)

    assert decisions[0].action == "apply_ocr_then_maybe_vision"
    assert decisions[0].reason == "ocr_first_for_visual_text"


def test_decision_directs_complex_formula_page_to_vision_when_enabled():
    decisions = build_enhancement_decisions([
        quality(5, text_score=0.25, reasons=["text_low", "visual_high", "formula_signal"]),
    ], vision_enabled=True)

    assert decisions[0].action == "apply_vision"
    assert decisions[0].reason == "visual_semantics_need_vision"
