from scholar_lens.parsers.models import ParsedPage
from scholar_lens.parsers.pdf_parser import diagnose_text_quality


def test_diagnose_text_quality_good_text_pdf():
    pages = [
        ParsedPage(page_num=0, text="a" * 1200, char_count=1200),
        ParsedPage(page_num=1, text="b" * 900, char_count=900),
        ParsedPage(page_num=2, text="c" * 700, char_count=700),
    ]

    result = diagnose_text_quality(pages, raw_text="x" * 2800, sections=[{"title": "Introduction"}, {"title": "Method"}, {"title": "Results"}])

    assert result["text_quality"] == "good"
    assert result["ocr_needed"] is False
    assert result["page_text_coverage"] == 1.0
    assert result["section_quality"] == "good"
    assert result["diagnostic_notes"] == []


def test_diagnose_text_quality_image_based_pdf():
    pages = [
        ParsedPage(page_num=0, text="", char_count=0),
        ParsedPage(page_num=1, text="tiny", char_count=4),
        ParsedPage(page_num=2, text="", char_count=0),
        ParsedPage(page_num=3, text="", char_count=0),
    ]

    result = diagnose_text_quality(pages, raw_text="tiny", sections=[])

    assert result["text_quality"] == "image_based"
    assert result["ocr_needed"] is True
    assert result["page_text_coverage"] == 0.0
    assert result["section_quality"] == "none"
    assert any("图片型" in note or "OCR" in note for note in result["diagnostic_notes"])


def test_diagnose_text_quality_weak_text_pdf():
    pages = [
        ParsedPage(page_num=0, text="a" * 120, char_count=120),
        ParsedPage(page_num=1, text="b" * 140, char_count=140),
    ]

    result = diagnose_text_quality(pages, raw_text="x" * 260, sections=[{"title": "Lecture 1"}])

    assert result["text_quality"] == "weak"
    assert result["ocr_needed"] is True
    assert result["section_quality"] == "weak"
