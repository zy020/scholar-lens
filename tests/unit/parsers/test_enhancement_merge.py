from scholar_lens.parsers.enhancement_merge import EnhancementFragment, merge_enhancements
from scholar_lens.parsers.models import ParsedDocument, ParsedPage


def test_merge_replaces_empty_page_text_with_good_ocr():
    doc = ParsedDocument(
        source_path="slides.pdf",
        parser_used="fake",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=2, text="", char_count=0)],
        raw_text="",
    )
    fragment = EnhancementFragment(page=2, source="ocr", text="Attention overview", quality="good")

    merged = merge_enhancements(doc, [fragment])

    assert merged.pages[0].text == "Attention overview"
    assert merged.pages[0].char_count == len("Attention overview")
    assert merged.pages[0].content_source == "ocr"
    assert merged.pages[0].enhanced is True
    assert merged.raw_text == "Attention overview"
    assert merged.sections[0]["id"] == "slide_2"
    assert merged.sections[0]["title"] == "Slide 3"
    assert merged.sections[0]["content_source"] == "ocr"
    assert merged.sections[0]["enhanced"] is True


def test_merge_appends_vision_text_to_existing_text():
    doc = ParsedDocument(
        source_path="slides.pdf",
        parser_used="fake",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=1, text="Existing", char_count=8)],
        raw_text="Existing",
    )
    fragment = EnhancementFragment(page=1, source="vision", text="Diagram explanation", quality="good")

    merged = merge_enhancements(doc, [fragment])

    assert "Existing" in merged.pages[0].text
    assert "[VISION]" in merged.pages[0].text
    assert merged.pages[0].content_source == "vision"
    assert merged.pages[0].enhanced is True
    assert "Diagram explanation" in merged.raw_text


def test_merge_ignores_failed_or_empty_fragments():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=0, text="Agenda", char_count=6)],
        raw_text="Agenda",
    )
    fragment = EnhancementFragment(page=0, source="ocr", text="", quality="failed")

    merged = merge_enhancements(doc, [fragment])

    assert merged.pages[0].text == "Agenda"
    assert merged.pages[0].content_source == "parser"
    assert merged.pages[0].enhanced is False
    assert merged.raw_text == "Agenda"


def test_merge_keeps_parser_when_ocr_candidate_is_lower_quality():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=1, text="Clear explanation of self-attention with queries keys and values.", char_count=62)],
        raw_text="Clear explanation of self-attention with queries keys and values.",
    )
    fragment = EnhancementFragment(page=1, source="ocr", text="Q K V ? ? | |", quality="weak")

    merged = merge_enhancements(doc, [fragment])

    assert merged.pages[0].text == "Clear explanation of self-attention with queries keys and values."
    assert merged.pages[0].content_source == "parser"
    assert merged.pages[0].enhanced is False


def test_merge_prefers_vision_over_ocr_for_visual_semantic_candidate():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=3, text="", char_count=0)],
        raw_text="",
    )
    ocr = EnhancementFragment(page=3, source="ocr", text="Q K V -> ->", quality="weak")
    vision = EnhancementFragment(page=3, source="vision", text="The diagram shows queries, keys, and values flowing into scaled dot-product attention.", quality="good")

    merged = merge_enhancements(doc, [ocr, vision])

    assert merged.pages[0].text == "The diagram shows queries, keys, and values flowing into scaled dot-product attention."
    assert merged.pages[0].content_source == "vision"
    assert merged.pages[0].enhanced is True


def test_vision_fragment_includes_structured_visual_fields_in_text():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=4, text="", char_count=0)],
        raw_text="",
    )
    fragment = EnhancementFragment(
        page=4,
        source="vision",
        text="The slide defines scaled dot-product attention.",
        quality="good",
        visual_type="formula",
        key_observations=["Q and K are multiplied"],
        formula_summary="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
        qa_hint="Useful for formula questions.",
    )

    merged = merge_enhancements(doc, [fragment])

    assert "Visual type: formula" in merged.pages[0].text
    assert "Formula summary: Attention" in merged.pages[0].text
    assert "QA hint: Useful for formula questions." in merged.pages[0].text


def test_vision_fragment_does_not_duplicate_existing_structured_fields():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=5, text="", char_count=0)],
        raw_text="",
    )
    fragment = EnhancementFragment(
        page=5,
        source="vision",
        text=(
            "The slide defines scaled dot-product attention.\n"
            "Visual type: formula\n"
            "Formula summary: Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V"
        ),
        quality="good",
        visual_type="formula",
        formula_summary="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
    )

    merged = merge_enhancements(doc, [fragment])

    assert merged.pages[0].text.count("Visual type: formula") == 1
    assert merged.pages[0].text.count("Formula summary:") == 1


def test_vision_fragment_with_only_structured_fields_is_usable():
    doc = ParsedDocument(
        source_path="slides.pdf",
        doc_subtype="slides_pdf",
        pages=[ParsedPage(page_num=6, text="", char_count=0)],
        raw_text="",
    )
    fragment = EnhancementFragment(
        page=6,
        source="vision",
        text="",
        quality="good",
        visual_type="table",
        table_summary="Rows compare Transformer and RNN training speed.",
    )

    merged = merge_enhancements(doc, [fragment])

    assert "Visual type: table" in merged.pages[0].text
    assert "Table summary: Rows compare Transformer" in merged.pages[0].text
