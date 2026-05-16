import pytest
from scholar_lens.parsers.models import ParsedPage, ParsedDocument, Chunk, ChunkMetadata


class TestParsedPage:
    def test_create(self):
        p = ParsedPage(page_num=1, text="Introduction text", char_count=17, is_two_column=False)
        assert p.page_num == 1
        assert p.char_count == 17

    def test_defaults(self):
        p = ParsedPage(page_num=0, text="")
        assert p.is_two_column is False


class TestParsedDocument:
    def test_create(self):
        doc = ParsedDocument(
            source_path="paper.pdf",
            doc_subtype="research_paper",
            pages=[ParsedPage(page_num=1, text="Abstract")],
            sections=[{"id": "1", "title": "Abstract", "level": 1, "text": "Abstract"}],
            raw_text="Abstract",
        )
        assert doc.doc_subtype == "research_paper"
        assert len(doc.pages) == 1

    def test_metadata(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[],
            sections=[],
            raw_text="",
        )
        assert doc.doc_subtype == "slides_pdf"


class TestChunkMetadata:
    def test_create(self):
        m = ChunkMetadata(
            section_id="3.1",
            section_type="method",
            chapter="3",
            difficulty_score=0.8,
            has_formula=False,
            cross_refs=["Fig 3"],
        )
        assert m.section_id == "3.1"
        assert m.cross_refs == ["Fig 3"]

    def test_defaults(self):
        m = ChunkMetadata(section_id="1", section_type="prose")
        assert m.difficulty_score == 0.5
        assert m.has_formula is False
        assert m.cross_refs == []


class TestChunk:
    def test_create(self):
        c = Chunk(
            chunk_id="paper_001_3.1_0",
            text="The self-attention mechanism computes...",
            metadata=ChunkMetadata(section_id="3.1", section_type="method"),
            layer="L2",
        )
        assert c.layer == "L2"

    def test_layers(self):
        for layer in ("L0", "L1", "L2"):
            c = Chunk(chunk_id="x", text="t", metadata=ChunkMetadata(section_id="1", section_type="prose"), layer=layer)
            assert c.layer == layer
