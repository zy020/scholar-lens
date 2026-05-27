import base64

import pytest
from scholar_lens.parsers.pdf_parser import detect_pdf_subtype, PDFParser
from scholar_lens.parsers.models import ParsedPage


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


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

    def test_no_signal_pdf_defaults_to_research_paper(self):
        pages = [
            ParsedPage(page_num=0, text="x" * 500, char_count=500, is_two_column=False),
        ]
        result = detect_pdf_subtype(pages)
        assert result == "research_paper"


class TestPDFParser:
    def test_instantiation(self):
        parser = PDFParser()
        assert parser is not None

    def test_parse_nonexistent_raises(self):
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.pdf")

    def test_parse_records_image_and_formula_metadata(self, tmp_path):
        import fitz

        pdf_path = tmp_path / "visual.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_image(fitz.Rect(20, 20, 220, 220), stream=PNG_1X1)
        page.insert_text((20, 260), "Scaled dot product: softmax(QK^T / sqrt(d_k)) V")
        doc.save(str(pdf_path))
        doc.close()

        parsed = PDFParser().parse(pdf_path)

        assert parsed.images
        assert parsed.images[0]["page"] == 0
        assert parsed.images[0]["area_ratio"] > 0
        assert parsed.formulas
        assert parsed.formulas[0]["page"] == 0
        assert "softmax" in parsed.formulas[0]["text"]

    def test_extract_toc_sections_converts_pymupdf_pages_to_zero_based(self):
        class FakeDoc:
            def get_toc(self, simple=False):
                return [
                    [1, "Introduction", 1, {}],
                    [1, "Method", 3, {}],
                ]

        sections = PDFParser._extract_toc_sections(FakeDoc())

        assert sections[0]["page_start"] == 0
        assert sections[1]["page_start"] == 2
