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
