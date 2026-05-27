from scholar_lens.parsers.models import ParsedDocument, ParsedPage
from scholar_lens.parsers.parse_quality import assess_parse_unit_quality, recommend_ocr_from_quality


def test_quality_good_for_text_page():
    doc = ParsedDocument(
        source_path="paper.pdf",
        doc_subtype="research_paper",
        pages=[ParsedPage(page_num=0, text="This page explains transformer attention. " * 40, char_count=1600)],
    )

    qualities = assess_parse_unit_quality(doc)

    assert qualities[0].quality == "good"
    assert qualities[0].recommended_action == "keep"
    assert qualities[0].text_score > 0.8


def test_quality_keeps_sparse_title_slide_without_visual_signal():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=0, text="Agenda", char_count=6)],
    )

    qualities = assess_parse_unit_quality(doc)

    assert qualities[0].recommended_action == "keep"
    assert "sparse_but_structured" in qualities[0].reasons


def test_quality_recommends_ocr_for_low_text_visual_page():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=2, text="", char_count=0)],
        images=[{"page": 2, "bbox": [0, 0, 500, 400]}],
    )

    qualities = assess_parse_unit_quality(doc)

    assert qualities[0].quality == "failed"
    assert qualities[0].recommended_action == "ocr"
    assert "text_low" in qualities[0].reasons
    assert "visual_high" in qualities[0].reasons


def test_quality_uses_table_and_formula_metadata_as_visual_signal():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[
            ParsedPage(page_num=1, text="", char_count=0),
            ParsedPage(page_num=2, text="softmax(QK^T / sqrt(d_k)) V", char_count=28),
        ],
        tables=[{"page": 1, "kind": "table"}],
        formulas=[{"page": 2, "text": "softmax(QK^T / sqrt(d_k)) V"}],
    )

    qualities = assess_parse_unit_quality(doc)

    assert qualities[0].recommended_action == "ocr"
    assert "visual_high" in qualities[0].reasons
    assert qualities[1].visual_score > 0
    assert qualities[1].recommended_action == "keep"


def test_quality_marks_low_text_formula_and_table_pages_for_visual_semantic_decision():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[
            ParsedPage(page_num=1, text="", char_count=0),
            ParsedPage(page_num=2, text="", char_count=0),
        ],
        formulas=[{"page": 1, "text": "large equation image"}],
        tables=[{"page": 2, "kind": "table"}],
    )

    qualities = assess_parse_unit_quality(doc)

    assert "formula_signal" in qualities[0].reasons
    assert "complex_table_signal" in qualities[1].reasons


def test_recommend_ocr_from_quality_keeps_existing_api_shape():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=2, text="", char_count=0)],
        images=[{"page": 2}],
    )

    recommendation = recommend_ocr_from_quality(assess_parse_unit_quality(doc))

    assert recommendation.pages == [2]
    assert recommendation.reasons["2"] == "text_low_parser_visuals"


def test_recommend_ocr_for_document_level_low_text_slides_without_visual_metadata():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[
            ParsedPage(page_num=0, text="Agenda", char_count=6),
            ParsedPage(page_num=1, text="Self-attention\nQ K V", char_count=20),
            ParsedPage(page_num=2, text="Attention matrix", char_count=16),
            ParsedPage(page_num=3, text="Summary", char_count=7),
        ],
    )

    recommendation = recommend_ocr_from_quality(assess_parse_unit_quality(doc))

    assert recommendation.pages == [1, 2]
    assert recommendation.reasons["1"] == "document_low_text_slides"
    assert recommendation.reasons["2"] == "document_low_text_slides"
