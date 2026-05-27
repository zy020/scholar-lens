import tempfile
from pathlib import Path

from scholar_lens.rag.document_store import DocumentStore
from scholar_lens.api.schemas import DocumentStatus, SectionSummary


class TestDocumentStore:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.store = DocumentStore(root=self.tmp)

    def test_create_writes_metadata(self):
        doc = self.store.create_document("test.pdf")
        assert doc.doc_id
        assert doc.name == "test.pdf"
        assert doc.status == DocumentStatus.uploaded
        assert doc.file_url == f"/api/documents/{doc.doc_id}/file"

    def test_save_load_sections_roundtrip(self):
        doc = self.store.create_document("paper.pdf")
        sections = [
            SectionSummary(section_id="1", title="Intro", level=1, page_start=1, gist="Overview"),
            SectionSummary(section_id="2", title="Method", level=1, page_start=3),
        ]
        self.store.save_sections(doc.doc_id, sections)
        loaded = self.store.load_sections(doc.doc_id)
        assert len(loaded) == 2
        assert loaded[0].title == "Intro"
        assert loaded[0].gist == "Overview"

        summary = self.store.get(doc.doc_id)
        assert summary.num_sections == 2

    def test_save_load_chunks(self):
        doc = self.store.create_document("paper.pdf")
        from scholar_lens.parsers.models import Chunk, ChunkMetadata
        chunks = [
            Chunk(chunk_id="c1", text="Hello world", metadata=ChunkMetadata(section_id="1", doc_id=doc.doc_id), layer="L2"),
            Chunk(chunk_id="c2", text="Goodbye", metadata=ChunkMetadata(section_id="2", doc_id=doc.doc_id), layer="L2"),
        ]
        self.store.save_chunks(doc.doc_id, chunks)
        loaded = self.store.load_chunks(doc.doc_id)
        assert len(loaded) == 2
        assert loaded[0]["text"] == "Hello world"

    def test_source_path_discovers_non_pdf_source(self):
        doc = self.store.create_document("slides.pptx")

        saved = self.store.save_source(doc.doc_id, b"pptx-bytes", suffix=".pptx")

        assert saved.name == "source.pptx"
        assert self.store.source_path(doc.doc_id).name == "source.pptx"
        assert self.store.source_path(doc.doc_id, suffix=".pptx") == saved

    def test_delete_removes_directory(self):
        doc = self.store.create_document("todel.pdf")
        self.store.delete(doc.doc_id)
        assert self.store.get(doc.doc_id) is None
        assert not self.store.document_dir(doc.doc_id).exists()

    def test_missing_document_returns_none(self):
        assert self.store.get("nonexistent") is None
        assert self.store.load_sections("nonexistent") == []
        assert self.store.load_chunks("nonexistent") == []

    def test_list_returns_summaries(self):
        self.store.create_document("a.pdf")
        self.store.create_document("b.pdf")
        docs = self.store.list()
        assert len(docs) == 2
        assert all(d.name.endswith(".pdf") for d in docs)

    def test_update_status(self):
        doc = self.store.create_document("x.pdf")
        self.store.update_status(doc.doc_id, DocumentStatus.failed, error="parse error")
        updated = self.store.get(doc.doc_id)
        assert updated.status == DocumentStatus.failed
        assert updated.error == "parse error"

    def test_save_and_load_document_understanding(self):
        from scholar_lens.core.models import DocumentUnderstanding, Section, Term

        doc = self.store.create_document("paper.pdf")
        understanding = DocumentUnderstanding(
            doc_type="research_paper",
            language="en",
            difficulty="intermediate",
            estimated_reading_time=12,
            sections=[Section(section_id="intro", title="Introduction", level=1)],
            mermaid_map="graph TD\n  doc-->intro",
            key_terms=[Term(english="RAG", chinese="检索增强生成")],
            l0_summaries={"intro": "Problem and motivation"},
            l1_overviews={"intro": "Longer overview"},
        )

        self.store.save_understanding(doc.doc_id, understanding)

        loaded = self.store.load_understanding(doc.doc_id)
        assert loaded is not None
        assert loaded.l0_summaries["intro"] == "Problem and motivation"
        assert loaded.key_terms[0].english == "RAG"
        assert loaded.mermaid_map.startswith("graph TD")

    def test_save_and_load_analysis_meta(self):
        doc = self.store.create_document("paper.pdf")

        self.store.save_analysis_meta(doc.doc_id, {
            "source": "llm",
            "updated_at": "2026-05-26T00:00:00Z",
            "error": "",
        })

        loaded = self.store.load_analysis_meta(doc.doc_id)
        assert loaded["source"] == "llm"
        assert loaded["updated_at"] == "2026-05-26T00:00:00Z"

    def test_save_and_load_parse_quality(self):
        from scholar_lens.parsers.parse_quality import ParseUnitQuality

        doc = self.store.create_document("slides.pdf")
        qualities = [
            ParseUnitQuality(
                unit_id="page_2",
                unit_type="slide",
                page_start=2,
                page_end=2,
                text_score=0.1,
                visual_score=0.75,
                quality="failed",
                recommended_action="ocr",
                reasons=["text_low", "visual_high"],
            )
        ]

        self.store.save_parse_quality(doc.doc_id, qualities)

        loaded = self.store.load_parse_quality(doc.doc_id)
        assert len(loaded) == 1
        assert loaded[0]["unit_id"] == "page_2"
        assert loaded[0]["recommended_action"] == "ocr"

    def test_save_and_load_ocr_enhancement(self):
        doc = self.store.create_document("slides.pdf")
        payload = {
            "doc_id": doc.doc_id,
            "status": "completed",
            "pages": [{"page": 2, "text": "OCR text", "ocr_quality": "good"}],
        }

        self.store.save_ocr_enhancement(doc.doc_id, payload)

        loaded = self.store.load_ocr_enhancement(doc.doc_id)
        assert loaded["status"] == "completed"
        assert loaded["pages"][0]["page"] == 2

    def test_save_and_load_parsed_document(self):
        from scholar_lens.parsers.models import ParsedDocument, ParsedPage

        doc = self.store.create_document("slides.pdf")
        parsed = ParsedDocument(
            source_path="slides.pdf",
            parser_used="fake",
            doc_subtype="slides_pdf",
            pages=[ParsedPage(page_num=1, text="Agenda", char_count=6)],
            raw_text="Agenda",
        )

        self.store.save_parsed_document(doc.doc_id, parsed)

        loaded = self.store.load_parsed_document(doc.doc_id)
        assert loaded is not None
        assert loaded.pages[0].page_num == 1
        assert loaded.raw_text == "Agenda"
