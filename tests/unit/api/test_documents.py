"""Tests for document routes per Ticket 3 acceptance criteria."""
import io
import pytest
import asyncio

pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.responses import FileResponse

from scholar_lens.api.main import create_app
from scholar_lens.api.routes import documents
from scholar_lens.api.schemas import DocumentStatus, SectionSummary
from scholar_lens.parsers.models import Chunk, ChunkMetadata, ParsedDocument
from scholar_lens.rag.document_store import DocumentStore
from tests.unit.api.helpers import ASGITestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "_store", DocumentStore(root=tmp_path))
    app = create_app()
    return ASGITestClient(app)


class TestDocumentRoutes:
    def test_list_empty(self, client):
        r = client.get("/api/documents")
        assert r.status_code == 200
        assert r.json()["docs"] == []

    def test_upload_rejects_non_pdf(self, client):
        r = client.post("/api/documents/upload", files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")})
        assert r.status_code == 415

    def test_upload_rejects_oversized(self, client, monkeypatch):
        monkeypatch.setattr(documents, "MAX_UPLOAD_SIZE_BYTES", 8)
        big = io.BytesIO(b"x" * 9)
        r = client.post("/api/documents/upload", files={"file": ("big.pdf", big, "application/pdf")})
        assert r.status_code == 413

    def test_upload_and_get(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="research_paper",
                    raw_text="A small parsed document.",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return [
                    Chunk(
                        chunk_id=f"{doc_id}_0_0",
                        text=parsed.raw_text,
                        metadata=ChunkMetadata(section_id="0", doc_id=doc_id),
                    )
                ]

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        pdf = io.BytesIO(b"%PDF-1.4 minimal pdf content")
        r = client.post("/api/documents/upload", files={"file": ("test.pdf", pdf, "application/pdf")})
        assert r.status_code == 200
        data = r.json()
        assert data["doc_id"]
        assert data["name"] == "test.pdf"
        assert data["status"] == "ready"
        assert data["file_url"].startswith("/api/documents/")

        doc_id = data["doc_id"]

        # List includes the document
        r2 = client.get("/api/documents")
        assert len(r2.json()["docs"]) == 1

        # Detail
        r3 = client.get(f"/api/documents/{doc_id}")
        assert r3.status_code == 200

        # FileResponse streaming hangs under httpx.ASGITransport in this env;
        # call the route directly to keep this a route contract test.
        file_response = asyncio.run(documents.get_file(doc_id))
        assert isinstance(file_response, FileResponse)
        assert file_response.media_type == "application/pdf"

        # Sections
        r5 = client.get(f"/api/documents/{doc_id}/sections")
        assert r5.status_code == 200
        assert "sections" in r5.json()

        # Delete
        r6 = client.delete(f"/api/documents/{doc_id}")
        assert r6.status_code == 200

    def test_upload_rejects_duplicate_filename(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="research_paper",
                    raw_text="A parsed document.",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return [
                    Chunk(
                        chunk_id=f"{doc_id}_0_0",
                        text=parsed.raw_text,
                        metadata=ChunkMetadata(section_id="0", doc_id=doc_id),
                    )
                ]

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        first = client.post(
            "/api/documents/upload",
            files={"file": ("same.pdf", io.BytesIO(b"%PDF-1.4 one"), "application/pdf")},
        )
        assert first.status_code == 200

        second = client.post(
            "/api/documents/upload",
            files={"file": ("same.pdf", io.BytesIO(b"%PDF-1.4 two"), "application/pdf")},
        )
        assert second.status_code == 409
        assert "already exists" in second.json()["detail"]

    def test_upload_rejects_duplicate_filename_before_reading_body(self, tmp_path, monkeypatch):
        store = DocumentStore(root=tmp_path)
        store.create_document("same.pdf")
        monkeypatch.setattr(documents, "_store", store)

        class ReadFailFile:
            filename = "same.pdf"

            async def read(self):
                raise AssertionError("duplicate upload should be rejected before reading content")

        with pytest.raises(HTTPException) as exc:
            asyncio.run(documents.upload_document(ReadFailFile()))

        assert exc.value.status_code == 409

    def test_get_section_text_returns_all_matching_chunks(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("paper.pdf")
        store.update_status(doc.doc_id, DocumentStatus.ready)
        store.save_sections(doc.doc_id, [
            SectionSummary(section_id="intro", title="Introduction", level=1),
            SectionSummary(section_id="method", title="Method", level=1),
        ])
        store.save_chunks(doc.doc_id, [
            Chunk(
                chunk_id="c1",
                text="Intro text should not be included.",
                metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
            ),
            Chunk(
                chunk_id="c2",
                text="First method paragraph.",
                metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
            ),
            Chunk(
                chunk_id="c3",
                text="Second method paragraph.",
                metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
            ),
        ])

        r = client.get(f"/api/documents/{doc.doc_id}/sections/method/text")

        assert r.status_code == 200
        data = r.json()
        assert data["section_id"] == "method"
        assert data["title"] == "Method"
        assert data["text"] == "First method paragraph.\n\nSecond method paragraph."
        assert data["num_chunks"] == 2

    def test_missing_doc_404(self, client):
        r = client.get("/api/documents/nonexist")
        assert r.status_code == 404

    def test_missing_file_404(self, client):
        r = client.get("/api/documents/nonexist/file")
        assert r.status_code == 404

    def test_upload_preserves_chunk_section_ids(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="research_paper",
                    sections=[
                        {"id": "intro", "title": "Introduction", "text": "Intro text."},
                        {"id": "method", "title": "Method", "text": "Method text."},
                    ],
                    raw_text="Intro text.\nMethod text.",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return [
                    Chunk(
                        chunk_id=f"{doc_id}_intro_0",
                        text="Intro first chunk.",
                        metadata=ChunkMetadata(section_id="intro", doc_id=doc_id),
                    ),
                    Chunk(
                        chunk_id=f"{doc_id}_intro_1",
                        text="Intro second chunk.",
                        metadata=ChunkMetadata(section_id="intro", doc_id=doc_id),
                    ),
                    Chunk(
                        chunk_id=f"{doc_id}_method_0",
                        text="Method chunk.",
                        metadata=ChunkMetadata(section_id="method", doc_id=doc_id),
                    ),
                ]

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        pdf = io.BytesIO(b"%PDF-1.4 fake pdf content")
        r = client.post("/api/documents/upload", files={"file": ("test.pdf", pdf, "application/pdf")})

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"

        chunks = documents._store.load_chunks(data["doc_id"])
        assert [c["metadata"]["section_id"] for c in chunks] == ["intro", "intro", "method"]

    def test_upload_stores_text_quality_diagnostics(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                from scholar_lens.parsers.models import ParsedPage
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="slides_pdf",
                    pages=[
                        ParsedPage(page_num=0, text="", char_count=0),
                        ParsedPage(page_num=1, text="", char_count=0),
                        ParsedPage(page_num=2, text="tiny", char_count=4),
                    ],
                    sections=[],
                    raw_text="tiny",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return []

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        r = client.post(
            "/api/documents/upload",
            files={"file": ("slides.pdf", io.BytesIO(b"%PDF-1.4 slides"), "application/pdf")},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["doc_type"] == "slides_pdf"
        assert data["text_quality"] == "image_based"
        assert data["ocr_needed"] is True
        assert data["page_text_coverage"] == 0.0
        assert data["section_quality"] == "none"
        assert data["diagnostic_notes"]
