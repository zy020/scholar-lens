"""Integration test: PDF parsing → chunking → vector store indexing flow."""

import pytest
from unittest.mock import MagicMock, patch
from scholar_lens.parsers.chunker import SectionAwareChunker
from scholar_lens.parsers.models import ParsedDocument, ParsedPage


class TestParseChunkIndexFlow:
    def _make_parsed_doc(self):
        return ParsedDocument(
            source_path="test.pdf",
            doc_subtype="research_paper",
            pages=[ParsedPage(page_num=i, text=f"Page {i} content. " * 50, char_count=900) for i in range(5)],
            sections=[
                {"id": "1", "title": "Introduction", "level": 1, "text": "This paper introduces a novel approach to machine translation using transformer architectures. " * 20},
                {"id": "2", "title": "Method", "level": 1, "text": "We propose a multi-head attention mechanism that allows the model to attend to different positions. " * 30},
                {"id": "3", "title": "Results", "level": 1, "text": "Experiments on WMT14 En-De show a BLEU score improvement of 2.0 points over the previous state of the art. " * 20},
            ],
            raw_text="Full paper text here...",
        )

    def test_parse_to_chunks_flow(self):
        """Verifies a parsed document gets properly chunked."""
        doc = self._make_parsed_doc()
        chunker = SectionAwareChunker(max_chunk_tokens=800)
        chunks = chunker.chunk(doc, doc_id="paper_001")
        assert len(chunks) >= 3
        section_ids = {chunk.metadata.section_id for chunk in chunks}
        assert "1" in section_ids
        assert "2" in section_ids
        assert "3" in section_ids

    def test_chunks_have_valid_ids(self):
        doc = self._make_parsed_doc()
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(doc, doc_id="p001")
        for chunk in chunks:
            assert chunk.chunk_id.startswith("p001_")
            assert chunk.text.strip()

    def test_chunks_preserve_metadata(self):
        doc = self._make_parsed_doc()
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(doc, doc_id="test")
        method_chunks = [c for c in chunks if c.metadata.section_id == "2"]
        assert len(method_chunks) > 0
        assert method_chunks[0].metadata.section_type == "method"

    def test_large_document_chunking(self):
        """Large documents should produce multiple chunks per section."""
        doc = ParsedDocument(
            source_path="large.pdf",
            doc_subtype="research_paper",
            sections=[
                {"id": "1", "title": "Long Section", "level": 1, "text": "A" * 5000},
            ],
            raw_text="A" * 5000,
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600)
        chunks = chunker.chunk(doc, doc_id="large")
        assert len(chunks) > 1
