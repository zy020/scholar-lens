from scholar_lens.parsers.models import ParsedDocument, ParsedPage
from scholar_lens.parsers.ocr_diagnostics import evaluate_ocr_result, recommend_ocr_pages


def test_sparse_blank_slide_is_not_recommended_for_ocr():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=0, text="Agenda", char_count=6)],
    )

    result = recommend_ocr_pages(doc)

    assert result.pages == []
    assert result.reasons == {}


def test_low_text_with_parser_visuals_is_recommended():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=2, text="", char_count=0)],
        images=[{"page": 2, "bbox": [0, 0, 500, 400]}],
    )

    result = recommend_ocr_pages(doc)

    assert result.pages == [2]
    assert result.reasons["2"] == "text_low_parser_visuals"


def test_visual_density_signal_recommends_page():
    doc = ParsedDocument(
        source_path="scan.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=3, text="tiny", char_count=4)],
    )

    result = recommend_ocr_pages(doc, visual_density_by_page={3: 0.42})

    assert result.pages == [3]
    assert result.reasons["3"] == "text_low_visual_high"


def test_many_zero_text_pages_escalate_image_based_document():
    doc = ParsedDocument(
        source_path="scan.pdf",
        doc_subtype="slides_pdf",
        pages=[
            ParsedPage(page_num=0, text="", char_count=0),
            ParsedPage(page_num=1, text="", char_count=0),
            ParsedPage(page_num=2, text="tiny", char_count=4),
            ParsedPage(page_num=3, text="", char_count=0),
        ],
    )

    result = recommend_ocr_pages(doc)

    assert result.pages == [0, 1, 2, 3]
    assert set(result.reasons.values()) == {"document_image_based"}


def test_many_sparse_title_slides_do_not_escalate_without_visual_evidence():
    doc = ParsedDocument(
        source_path="titles.pdf",
        doc_subtype="slides_pdf",
        pages=[
            ParsedPage(page_num=0, text="Agenda", char_count=6),
            ParsedPage(page_num=1, text="Motivation", char_count=10),
            ParsedPage(page_num=2, text="Summary", char_count=7),
            ParsedPage(page_num=3, text="Q&A", char_count=3),
        ],
    )

    result = recommend_ocr_pages(doc)

    assert result.pages == []
    assert result.reasons == {}


def test_evaluate_ocr_result_good_text_solves_page():
    result = evaluate_ocr_result(
        ocr_text="Transformer attention computes contextual token representations for each position.",
        visual_density=0.1,
    )

    assert result.ocr_quality == "good"
    assert result.vision_recommended is False
    assert result.reason == "ocr_text_usable"


def test_evaluate_ocr_result_short_text_recommends_vision():
    result = evaluate_ocr_result(ocr_text="Q K V", visual_density=0.35)

    assert result.ocr_quality == "weak"
    assert result.vision_recommended is True
    assert result.reason == "ocr_too_short_visual_high"


def test_evaluate_ocr_result_garbled_text_failed():
    result = evaluate_ocr_result(ocr_text="|||| ____ #### ////", visual_density=0.05)

    assert result.ocr_quality == "failed"
    assert result.vision_recommended is True
    assert result.reason == "garbled_text"


def test_evaluate_ocr_result_diagram_like_page_recommends_vision_even_with_labels():
    result = evaluate_ocr_result(
        ocr_text="Input Layer Output Attention Scores",
        visual_density=0.42,
        page_kind="diagram",
    )

    assert result.ocr_quality == "weak"
    assert result.vision_recommended is True
    assert result.reason == "diagram_like"
