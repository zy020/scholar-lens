import pytest
from scholar_lens.parsers.chunker import SectionAwareChunker
from scholar_lens.parsers.models import ParsedDocument, Chunk


class TestSectionAwareChunker:
    def _make_doc(self, sections=None, raw_text=""):
        return ParsedDocument(source_path="test.pdf", doc_subtype="research_paper", sections=sections or [], raw_text=raw_text)

    def test_chunk_by_section(self):
        doc = self._make_doc(
            sections=[
                {"id": "1", "title": "Intro", "level": 1, "text": "A" * 500},
                {"id": "2", "title": "Method", "level": 1, "text": "B" * 500},
            ],
            raw_text="A" * 500 + "\n" + "B" * 500,
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="test_doc")
        assert len(chunks) >= 2
        assert chunks[0].metadata.section_id == "1"
        assert chunks[1].metadata.section_id == "2"

    def test_large_section_split(self):
        doc = self._make_doc(sections=[{"id": "1", "title": "Long", "level": 1, "text": "X" * 3000}], raw_text="X" * 3000)
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="test_doc")
        assert len(chunks) > 1

    def test_chunk_metadata_populated(self):
        doc = self._make_doc(sections=[{"id": "3.1", "title": "Model", "level": 2, "text": "Short text about the model."}], raw_text="Short text about the model.")
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="paper_001")
        assert len(chunks) == 1
        assert chunks[0].metadata.section_id == "3.1"
        assert chunks[0].metadata.doc_id == "paper_001"
        assert chunks[0].layer == "L2"

    def test_empty_document(self):
        doc = self._make_doc(raw_text="")
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="empty")
        assert chunks == []

    def test_chunk_id_format(self):
        doc = self._make_doc(sections=[{"id": "1", "title": "A", "level": 1, "text": "Hello world"}], raw_text="Hello world")
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="p001")
        assert chunks[0].chunk_id.startswith("p001_1_")

    def test_references_section_skipped(self):
        doc = self._make_doc(
            sections=[
                {"id": "1", "title": "Intro", "level": 1, "text": "Some intro text"},
                {"id": "ref", "title": "References", "level": 1, "text": "[1] Smith. Paper. 2020."},
            ],
            raw_text="Some intro text\n\nReferences\n[1] Smith. Paper. 2020.",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="test")
        assert len(chunks) == 1
        assert chunks[0].metadata.section_id == "1"
