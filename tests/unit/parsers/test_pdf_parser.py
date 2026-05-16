import pytest
from scholar_lens.parsers.pdf_parser import detect_pdf_subtype, PDFParser
from scholar_lens.parsers.models import ParsedPage


class TestDetectPdfSubtype:
    def test_research_paper_detected(self):
        pages = [
            ParsedPage(page_num=0, text="Abstract\n" + "x" * 2000, char_count=2010, is_two_column=True, has_abstract=True),
            ParsedPage(page_num=1, text="y" * 3000, char_count=3000, is_two_column=True),
        ]
        result = detect_pdf_subtype(pages)
        assert result == "research_paper"

    def test_slides_pdf_detected(self):
        pages = [
            ParsedPage(page_num=0, text="Title", char_count=50, is_two_column=False),
            ParsedPage(page_num=1, text="Bullet", char_count=100, is_two_column=False),
        ]
        result = detect_pdf_subtype(pages)
        assert result == "slides_pdf"

    def test_general_document(self):
        pages = [
            ParsedPage(page_num=0, text="x" * 500, char_count=500, is_two_column=False),
        ]
        result = detect_pdf_subtype(pages)
        assert result == "general_document"


class TestPDFParser:
    def test_instantiation(self):
        parser = PDFParser()
        assert parser is not None

    def test_parse_nonexistent_raises(self):
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.pdf")
