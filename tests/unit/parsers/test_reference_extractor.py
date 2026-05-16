import pytest
from scholar_lens.parsers.reference_extractor import ReferenceExtractor, ReferenceExtraction


class TestReferenceExtractor:
    def test_extract_references_section(self):
        text = (
            "1. Introduction\n"
            "We build on prior work [3].\n\n"
            "References\n"
            "[1] Vaswani, A. et al. Attention Is All You Need. NeurIPS 2017.\n"
            "[2] Devlin, J. et al. BERT. NAACL 2019.\n"
            "[3] Brown, T. et al. GPT-3. NeurIPS 2020.\n"
        )
        extractor = ReferenceExtractor()
        result = extractor.extract(text)
        assert len(result.references) == 3
        assert result.references[0].ref_id == "1"

    def test_citation_contexts_preserved(self):
        text = "We extend [3] to handle multi-modal inputs.\nReferences\n[3] Brown et al. GPT-3. 2020.\n"
        extractor = ReferenceExtractor()
        result = extractor.extract(text)
        assert len(result.citation_contexts) >= 1
        assert "3" in result.citation_contexts[0].ref_id

    def test_no_references_section(self):
        text = "Just some text without any references."
        extractor = ReferenceExtractor()
        result = extractor.extract(text)
        assert len(result.references) == 0
        assert len(result.citation_contexts) == 0

    def test_body_text_without_references(self):
        text = "Introduction text here.\nReferences\n[1] Smith. Paper. 2020.\n"
        extractor = ReferenceExtractor()
        result = extractor.extract(text)
        assert "References" not in result.body_text
        assert "Introduction text here" in result.body_text
