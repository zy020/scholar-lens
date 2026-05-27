"""Tests for document routes per Ticket 3 acceptance criteria."""
import io
import pytest
import asyncio
from unittest.mock import AsyncMock

pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.responses import FileResponse

from scholar_lens.api.main import create_app
from scholar_lens.api.routes import documents
from scholar_lens.api.document_analysis import AnalysisRunResult
from scholar_lens.api.schemas import DocumentStatus, SectionSummary
from scholar_lens.core.settings import Settings
from scholar_lens.parsers.models import Chunk, ChunkMetadata, ParsedDocument
from scholar_lens.parsers.parse_quality import ParseUnitQuality
from scholar_lens.rag.document_store import DocumentStore
from tests.unit.api.helpers import ASGITestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "_store", DocumentStore(root=tmp_path))
    monkeypatch.setattr(
        documents,
        "get_settings",
        lambda: Settings(
            _env_file="",
            api_key="",
            auto_ocr_enabled=False,
            llm_quality_enabled=False,
            vision_enhancement_enabled=False,
        ),
    )
    monkeypatch.setattr(documents, "index_document_chunks", lambda *args, **kwargs: False, raising=False)
    app = create_app()
    return ASGITestClient(app)


class TestDocumentRoutes:
    def test_list_empty(self, client):
        r = client.get("/api/documents")
        assert r.status_code == 200
        assert r.json()["docs"] == []

    def test_upload_rejects_unsupported_extension(self, client):
        r = client.post(
            "/api/documents/upload/paper",
            files={"file": ("test.docx", io.BytesIO(b"hello"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert r.status_code == 415

    def test_upload_rejects_oversized(self, client, monkeypatch):
        monkeypatch.setattr(documents, "MAX_UPLOAD_SIZE_BYTES", 8)
        big = io.BytesIO(b"x" * 9)
        r = client.post("/api/documents/upload/paper", files={"file": ("big.pdf", big, "application/pdf")})
        assert r.status_code == 413

    def test_upload_paper_endpoint_forces_research_paper(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="slides_pdf",
                    raw_text="A paper-like parsed document.",
                )

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)

        r = client.post(
            "/api/documents/upload/paper",
            files={"file": ("paper.pdf", io.BytesIO(b"%PDF paper"), "application/pdf")},
        )

        assert r.status_code == 200
        assert r.json()["doc_type"] == "research_paper"

    def test_upload_courseware_endpoint_forces_pdf_to_slides(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="research_paper",
                    raw_text="Slide 1\nSelf-attention",
                )

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)

        r = client.post(
            "/api/documents/upload/courseware",
            files={"file": ("slides.pdf", io.BytesIO(b"%PDF slides"), "application/pdf")},
        )

        assert r.status_code == 200
        assert r.json()["doc_type"] == "slides_pdf"

    def test_upload_courseware_endpoint_accepts_pptx(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="research_paper",
                    raw_text="Slide 1\nSelf-attention",
                )

        monkeypatch.setattr("scholar_lens.parsers.ppt_parser.PPTParser", FakeParser)

        r = client.post(
            "/api/documents/upload/courseware",
            files={"file": ("slides.pptx", io.BytesIO(b"pptx"), "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
        )

        assert r.status_code == 200
        assert r.json()["doc_type"] == "courseware_pptx"

    def test_upload_paper_endpoint_rejects_pptx(self, client):
        r = client.post(
            "/api/documents/upload/paper",
            files={"file": ("slides.pptx", io.BytesIO(b"pptx"), "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
        )

        assert r.status_code == 415

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
        r = client.post("/api/documents/upload/paper", files={"file": ("test.pdf", pdf, "application/pdf")})
        assert r.status_code == 200
        data = r.json()
        assert data["doc_id"]
        assert data["name"] == "test.pdf"
        assert data["status"] == "ready"
        assert data["index_status"] == "keyword_only"
        assert data["file_url"].startswith("/api/documents/")
        assert documents._store.load_understanding(data["doc_id"]) is None

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

    def test_upload_attempts_vector_indexing_after_chunks_saved(self, client, monkeypatch):
        calls = []

        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="research_paper",
                    raw_text="A small parsed document.",
                )

        class FakeChunker:
            def __init__(self, *args, **kwargs):
                pass

            def chunk(self, parsed, doc_id=""):
                return [
                    Chunk(
                        chunk_id=f"{doc_id}_0_0",
                        text=parsed.raw_text,
                        metadata=ChunkMetadata(section_id="0", doc_id=doc_id),
                    )
                ]

        def fake_index(store, doc_id, chunks, settings):
            calls.append((store, doc_id, [chunk.chunk_id for chunk in chunks], settings))
            return True

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)
        monkeypatch.setattr(documents, "index_document_chunks", fake_index, raising=False)

        r = client.post(
            "/api/documents/upload/paper",
            files={"file": ("indexed.pdf", io.BytesIO(b"%PDF-1.4 indexed"), "application/pdf")},
        )

        assert r.status_code == 200
        assert r.json()["index_status"] == "vector"
        assert len(calls) == 1
        assert calls[0][0] is documents._store
        assert calls[0][1] == r.json()["doc_id"]
        assert calls[0][2] == [f"{r.json()['doc_id']}_0_0"]

    def test_upload_accepts_pptx_and_preserves_slide_chunks(self, client, monkeypatch):
        monkeypatch.setattr(documents, "index_document_chunks", lambda *args, **kwargs: False, raising=False)
        monkeypatch.setattr(documents, "_auto_enhance_after_upload", AsyncMock(return_value=None))

        class FakePPTParser:
            def parse(self, source):
                assert source.name == "source.pptx"
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake-pptx",
                    doc_subtype="courseware_pptx",
                    sections=[
                        {"id": "slide_0", "title": "Intro", "text": "Intro\nCourse goals."},
                        {"id": "slide_1", "title": "Attention", "text": "Attention\nQ K V."},
                    ],
                    raw_text="Intro\n\nAttention",
                )

        monkeypatch.setattr("scholar_lens.parsers.ppt_parser.PPTParser", FakePPTParser)

        r = client.post(
            "/api/documents/upload/courseware",
            files={
                "file": (
                    "lecture.pptx",
                    io.BytesIO(b"pptx bytes"),
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        assert data["doc_type"] == "courseware_pptx"
        assert documents._store.source_path(data["doc_id"]).name == "source.pptx"
        chunks = documents._store.load_chunks(data["doc_id"])
        assert [c["metadata"]["content_type"] for c in chunks] == ["slide", "slide"]
        sections = documents._store.load_sections(data["doc_id"])
        assert [section.title for section in sections] == ["Slide 1", "Slide 2"]

    def test_section_text_falls_back_to_parsed_slide_text(self, client):
        store = documents._store
        doc = store.create_document("lecture.pptx")
        store.update_status(doc.doc_id, DocumentStatus.ready)
        store.save_sections(doc.doc_id, [
            SectionSummary(section_id="slide_0", title="Slide 1", level=1, gist="Short gist."),
        ])
        store.save_chunks(doc.doc_id, [])
        store.save_parsed_document(doc.doc_id, ParsedDocument(
            source_path="lecture.pptx",
            parser_used="fake",
            doc_subtype="courseware_pptx",
            sections=[{"id": "slide_0", "title": "Outline", "text": "Outline\nCourse goals and attention."}],
            raw_text="Outline\nCourse goals and attention.",
        ))

        r = client.get(f"/api/documents/{doc.doc_id}/sections/slide_0/text")

        assert r.status_code == 200
        assert r.json()["text"] == "Outline\nCourse goals and attention."

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
            "/api/documents/upload/paper",
            files={"file": ("same.pdf", io.BytesIO(b"%PDF-1.4 one"), "application/pdf")},
        )
        assert first.status_code == 200

        second = client.post(
            "/api/documents/upload/paper",
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
            asyncio.run(documents.upload_paper_document(ReadFailFile()))

        assert exc.value.status_code == 409

    def test_analyze_endpoint_enhances_document(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("paper.pdf")
        store.update_status(doc.doc_id, DocumentStatus.ready)

        async def fake_enhance_document_analysis(store_arg, doc_id, settings=None, memory_manager=None):
            assert store_arg is store
            assert doc_id == doc.doc_id
            return AnalysisRunResult(doc_id=doc_id, status="enhanced", source="llm")

        monkeypatch.setattr(documents, "enhance_document_analysis", fake_enhance_document_analysis)

        r = client.post(f"/api/documents/{doc.doc_id}/analyze")

        assert r.status_code == 200
        assert r.json()["doc_id"] == doc.doc_id
        assert r.json()["status"] == "enhanced"
        assert r.json()["source"] == "llm"

    def test_analyze_endpoint_404_for_missing_doc(self, client):
        r = client.post("/api/documents/missing/analyze")

        assert r.status_code == 404

    def test_enhance_plan_returns_stored_ocr_recommendations(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.update_summary(
            doc.doc_id,
            ocr_recommended_pages=[2, 4],
            ocr_recommendation_reasons={
                "2": "text_low_parser_visuals",
                "4": "document_image_based",
            },
        )

        r = client.post(f"/api/documents/{doc.doc_id}/enhance-plan")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "planned"
        assert data["recommended_ocr_pages"] == [2, 4]
        assert data["estimated_ocr_pages"] == 2
        assert data["vision_escalation_reasons"]

    def test_enhance_plan_skips_when_no_ocr_pages(self, client):
        store = documents._store
        doc = store.create_document("paper.pdf")

        r = client.post(f"/api/documents/{doc.doc_id}/enhance-plan")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "skipped"
        assert data["estimated_ocr_pages"] == 0

    def test_enhance_plan_reports_vision_availability_from_settings(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[1])

        class FakeSettings:
            vision_model = "vision-model"
            vision_api_key = "vision-key"
            vision_base_url = "https://vision.example/v1"
            vision_enhancement_enabled = True

        monkeypatch.setattr(documents, "get_settings", lambda: FakeSettings())

        r = client.post(f"/api/documents/{doc.doc_id}/enhance-plan")

        assert r.status_code == 200
        data = r.json()
        assert data["vision_available"] is True
        assert data["vision_enhancement_enabled"] is True
        assert data["vision_possible"] is True

    def test_enhance_plan_requires_enabled_vision_for_manual_action(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[1])

        class FakeSettings:
            vision_model = "vision-model"
            vision_api_key = "vision-key"
            vision_base_url = "https://vision.example/v1"
            vision_enhancement_enabled = False

        monkeypatch.setattr(documents, "get_settings", lambda: FakeSettings())

        r = client.post(f"/api/documents/{doc.doc_id}/enhance-plan")

        assert r.status_code == 200
        data = r.json()
        assert data["vision_available"] is True
        assert data["vision_enhancement_enabled"] is False
        assert data["vision_possible"] is False

    def test_enhance_plan_includes_ocr_capability(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[1])

        class FakeCapability:
            engine = "rapidocr"
            installed = True
            gpu_available = False
            cpu_available = True
            recommended_mode = "ask_user"
            available_actions = ["cpu_ocr", "vision"]

        monkeypatch.setattr(documents, "detect_rapidocr_capability", lambda vision_available=False: FakeCapability())

        r = client.post(f"/api/documents/{doc.doc_id}/enhance-plan")

        assert r.status_code == 200
        data = r.json()
        assert data["ocr_engine"] == "rapidocr"
        assert data["ocr_installed"] is True
        assert data["ocr_gpu_available"] is False
        assert data["ocr_cpu_available"] is True
        assert data["ocr_recommended_mode"] == "ask_user"
        assert data["available_actions"] == ["cpu_ocr", "vision"]

    def test_enhance_plan_includes_page_decisions_from_quality_and_ocr_probe(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[2])
        store.save_parse_quality(doc.doc_id, [
            ParseUnitQuality(
                unit_id="page_2",
                unit_type="slide",
                page_start=2,
                page_end=2,
                text_score=0.1,
                structure_score=0.3,
                visual_score=0.7,
                ocr_need_score=0.8,
                overall_score=0.2,
                quality="failed",
                recommended_action="ocr",
                reasons=["text_low", "visual_high"],
            )
        ])
        store.save_ocr_enhancement(doc.doc_id, {
            "pages": [
                {"page": 2, "text": "Self-attention maps queries to keys and values. " * 3, "ocr_quality": "good"},
            ]
        })

        r = client.post(f"/api/documents/{doc.doc_id}/enhance-plan")

        assert r.status_code == 200
        data = r.json()
        assert data["page_decisions"][0]["page"] == 2
        assert data["page_decisions"][0]["action"] == "apply_ocr"
        assert data["page_decisions"][0]["reason"] == "ocr_readable_gain"

    def test_enhance_plan_does_not_fallback_to_stale_ocr_pages_when_quality_keeps_page(self, client):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[1])
        store.save_parse_quality(doc.doc_id, [
            ParseUnitQuality(
                unit_id="page_1",
                unit_type="slide",
                page_start=1,
                page_end=1,
                text_score=0.1,
                structure_score=0.4,
                visual_score=0.0,
                ocr_need_score=0.0,
                overall_score=0.3,
                quality="failed",
                recommended_action="keep",
                reasons=["text_low", "sparse_but_structured"],
            )
        ])

        r = client.post(f"/api/documents/{doc.doc_id}/enhance-plan")

        assert r.status_code == 200
        data = r.json()
        assert data["recommended_ocr_pages"] == []
        assert data["status"] == "skipped"
        assert data["page_decisions"][0]["action"] == "use_original"

    def test_enhance_plan_404_for_missing_doc(self, client):
        r = client.post("/api/documents/missing/enhance-plan")

        assert r.status_code == 404

    def test_ocr_enhance_skips_when_no_pages(self, client):
        store = documents._store
        doc = store.create_document("paper.pdf")

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/ocr")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "skipped"
        assert data["pages"] == []

    def test_ocr_enhance_runs_executor_and_persists_result(self, client, monkeypatch):
        from scholar_lens.api.schemas import OCREnhanceResponse

        store = documents._store
        doc = store.create_document("slides.pdf")
        store.save_source(doc.doc_id, b"%PDF-1.4", suffix=".pdf")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[2])

        async def fake_runner(store_arg, doc_id, mode="auto"):
            assert store_arg is store
            return OCREnhanceResponse(
                doc_id=doc_id,
                status="completed",
                pages=[{"page": 2, "text": "OCR text", "ocr_quality": "good"}],
            )

        monkeypatch.setattr(documents, "run_rapidocr_enhancement", fake_runner)

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/ocr?mode=cpu")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "completed"
        assert data["pages"][0]["text"] == "OCR text"
        saved = store.load_ocr_enhancement(doc.doc_id)
        assert saved["pages"][0]["page"] == 2

    def test_ocr_enhance_runs_for_pptx_and_persists_result(self, client, monkeypatch):
        from scholar_lens.api.schemas import OCREnhanceResponse

        store = documents._store
        doc = store.create_document("slides.pptx", suffix=".pptx")
        store.save_source(doc.doc_id, b"pptx", suffix=".pptx")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[1])

        async def fake_runner(store_arg, doc_id, mode="auto"):
            assert store_arg is store
            return OCREnhanceResponse(
                doc_id=doc_id,
                status="completed",
                pages=[{"page": 1, "text": "Rendered PPTX OCR text", "ocr_quality": "good"}],
            )

        monkeypatch.setattr(documents, "run_rapidocr_enhancement", fake_runner)

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/ocr")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "completed"
        assert data["pages"][0]["text"] == "Rendered PPTX OCR text"
        saved = store.load_ocr_enhancement(doc.doc_id)
        assert saved["pages"][0]["page"] == 1

    def test_vision_enhance_runs_for_recommended_pages_and_persists_result(self, client, monkeypatch):
        from scholar_lens.api.schemas import VisionEnhanceResponse

        store = documents._store
        doc = store.create_document("slides.pdf")
        store.save_source(doc.doc_id, b"%PDF-1.4", suffix=".pdf")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[2])
        store.save_ocr_enhancement(doc.doc_id, {
            "doc_id": doc.doc_id,
            "status": "completed",
            "vision_recommended_pages": [2],
            "pages": [{"page": 2, "text": "", "ocr_quality": "failed", "vision_recommended": True}],
        })

        class FakeSettings:
            vision_api_key = "vision-key"
            vision_base_url = "https://vision.example/v1"
            vision_model = "vision-model"

        async def fake_runner(store_arg, doc_id, pages, settings):
            assert store_arg is store
            assert pages == [2]
            assert settings is FakeSettings
            return VisionEnhanceResponse(
                doc_id=doc_id,
                status="completed",
                pages=[{"page": 2, "text": "Vision diagram explanation", "vision_quality": "good"}],
            )

        monkeypatch.setattr(documents, "get_settings", lambda: FakeSettings)
        monkeypatch.setattr(documents, "run_vision_enhancement", fake_runner)

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/vision")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "completed"
        assert data["pages"][0]["text"] == "Vision diagram explanation"
        saved = store.load_vision_enhancement(doc.doc_id)
        assert saved["pages"][0]["page"] == 2

    def test_vision_enhance_requires_enabled_policy(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.save_source(doc.doc_id, b"%PDF-1.4", suffix=".pdf")
        store.update_summary(doc.doc_id, ocr_recommended_pages=[2])

        class FakeSettings:
            vision_api_key = "vision-key"
            vision_base_url = "https://vision.example/v1"
            vision_model = "vision-model"
            vision_enhancement_enabled = False

        monkeypatch.setattr(documents, "get_settings", lambda: FakeSettings)

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/vision")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "unavailable"
        assert "not enabled" in data["error"]

    def test_apply_enhancement_requires_ocr_payload(self, client):
        store = documents._store
        doc = store.create_document("slides.pdf")

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/apply")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "missing"
        assert "OCR" in data["message"]

    def test_apply_enhancement_merges_ocr_and_rechunks(self, client):
        from scholar_lens.parsers.models import ParsedDocument, ParsedPage

        store = documents._store
        doc = store.create_document("slides.pdf")
        parsed = ParsedDocument(
            source_path="slides.pdf",
            parser_used="fake",
            doc_subtype="slides_pdf",
            pages=[ParsedPage(page_num=2, text="", char_count=0)],
            raw_text="",
        )
        store.save_parsed_document(doc.doc_id, parsed)
        store.save_ocr_enhancement(doc.doc_id, {
            "doc_id": doc.doc_id,
            "status": "completed",
            "pages": [
                {
                    "page": 2,
                    "text": "Attention overview with enough OCR text for chunking.",
                    "ocr_quality": "good",
                    "vision_recommended": False,
                }
            ],
        })

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/apply")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "applied"
        assert data["num_pages_updated"] == 1
        enhanced = store.load_parsed_document(doc.doc_id, enhanced=True)
        assert "Attention overview" in enhanced.raw_text
        chunks = store.load_chunks(doc.doc_id)
        assert chunks
        assert "Attention overview" in chunks[0]["text"]
        qualities = store.load_parse_quality(doc.doc_id)
        assert qualities[0]["recommended_action"] == "keep"

    def test_apply_enhancement_merges_vision_payload_without_ocr_payload(self, client):
        from scholar_lens.parsers.models import ParsedDocument, ParsedPage

        store = documents._store
        doc = store.create_document("slides.pdf")
        parsed = ParsedDocument(
            source_path="slides.pdf",
            parser_used="fake",
            doc_subtype="slides_pdf",
            pages=[ParsedPage(page_num=1, text="Existing caption", char_count=16)],
            raw_text="Existing caption",
        )
        store.save_parsed_document(doc.doc_id, parsed)
        store.save_vision_enhancement(doc.doc_id, {
            "doc_id": doc.doc_id,
            "status": "completed",
            "pages": [{"page": 1, "text": "Vision explains the diagram.", "vision_quality": "good"}],
        })

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/apply")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "applied"
        enhanced = store.load_parsed_document(doc.doc_id, enhanced=True)
        assert "[VISION]" in enhanced.raw_text
        assert "Vision explains the diagram." in enhanced.raw_text

    def test_apply_enhancement_attempts_vector_reindexing(self, client, monkeypatch):
        from scholar_lens.parsers.models import ParsedDocument, ParsedPage

        calls = []

        def fake_index(store, doc_id, chunks, settings):
            calls.append((store, doc_id, [chunk.chunk_id for chunk in chunks], settings))
            return True

        monkeypatch.setattr(documents, "index_document_chunks", fake_index, raising=False)
        store = documents._store
        doc = store.create_document("slides.pdf")
        parsed = ParsedDocument(
            source_path="slides.pdf",
            parser_used="fake",
            doc_subtype="slides_pdf",
            pages=[ParsedPage(page_num=2, text="", char_count=0)],
            raw_text="",
        )
        store.save_parsed_document(doc.doc_id, parsed)
        store.save_ocr_enhancement(doc.doc_id, {
            "doc_id": doc.doc_id,
            "status": "completed",
            "pages": [{"page": 2, "text": "Attention overview with enough OCR text for chunking.", "ocr_quality": "good"}],
        })

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/apply")

        assert r.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] is store
        assert calls[0][1] == doc.doc_id
        assert calls[0][2]

    def test_quality_endpoint_returns_heuristic_records(self, client):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.save_parse_quality(doc.doc_id, [
            {"unit_id": "page_1", "quality": "good", "recommended_action": "keep"},
        ])

        r = client.get(f"/api/documents/{doc.doc_id}/quality")

        assert r.status_code == 200
        data = r.json()
        assert data["doc_id"] == doc.doc_id
        assert data["source"] == "heuristic"
        assert data["status"] == "available"
        assert data["qualities"][0]["unit_id"] == "page_1"

    def test_quality_endpoint_uses_llm_only_when_requested(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.save_parse_quality(doc.doc_id, [
            {"unit_id": "page_1", "quality": "weak", "recommended_action": "keep"},
        ])
        called = {"count": 0}

        async def fake_llm(store_arg, doc_id, settings=None):
            from scholar_lens.api.schemas import ParseQualityResponse

            assert store_arg is store
            called["count"] += 1
            return ParseQualityResponse(
                doc_id=doc_id,
                source="llm",
                status="available",
                qualities=[{"unit_id": "page_1", "quality": "good"}],
            )

        monkeypatch.setattr(documents, "evaluate_parse_quality_with_llm", fake_llm)

        heuristic = client.post(f"/api/documents/{doc.doc_id}/quality/evaluate")

        assert heuristic.status_code == 200
        assert heuristic.json()["source"] == "heuristic"
        assert called["count"] == 0

        llm = client.post(f"/api/documents/{doc.doc_id}/quality/evaluate?use_llm=true")

        assert llm.status_code == 200
        assert llm.json()["source"] == "llm"
        assert called["count"] == 1

    def test_llm_quality_evaluation_returns_unavailable_without_model_config(self, client):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.save_parse_quality(doc.doc_id, [
            {"unit_id": "page_1", "quality": "weak", "recommended_action": "keep"},
        ])

        class FakeSettings:
            llm_api_key = ""
            llm_model = ""

        result = asyncio.run(documents.evaluate_parse_quality_with_llm(store, doc.doc_id, settings=FakeSettings()))

        assert result.status == "unavailable"
        assert result.qualities[0]["unit_id"] == "page_1"

    def test_llm_quality_evaluation_merges_model_json(self, client, monkeypatch):
        store = documents._store
        doc = store.create_document("slides.pdf")
        store.save_parse_quality(doc.doc_id, [
            {"unit_id": "page_1", "quality": "weak", "recommended_action": "keep"},
        ])

        class FakeSettings:
            llm_api_key = "test-api-key"
            llm_model = "fake-model"

        class FakeResponse:
            content = '{"qualities":[{"unit_id":"page_1","llm_score":0.91,"quality":"good","recommended_action":"keep","llm_reason":"usable text"}]}'

        class FakeLLM:
            async def ainvoke(self, messages):
                assert "heuristic_qualities" in messages[0].content
                return FakeResponse()

        class FakeFactory:
            def create(self, streaming=False):
                assert streaming is False
                return FakeLLM()

        monkeypatch.setattr(
            "scholar_lens.core.llm_factory.ChatLLMFactory.from_settings",
            lambda settings: FakeFactory(),
        )

        result = asyncio.run(documents.evaluate_parse_quality_with_llm(store, doc.doc_id, settings=FakeSettings()))

        assert result.status == "available"
        assert result.qualities[0]["quality"] == "good"
        assert result.qualities[0]["llm_score"] == 0.91
        assert result.qualities[0]["llm_reason"] == "usable text"

    def test_analysis_endpoint_returns_missing_for_doc_without_analysis(self, client):
        store = documents._store
        doc = store.create_document("paper.pdf")
        store.update_status(doc.doc_id, DocumentStatus.ready)

        r = client.get(f"/api/documents/{doc.doc_id}/analysis")

        assert r.status_code == 200
        assert r.json()["status"] == "missing"
        assert r.json()["source"] == "missing"

    def test_analysis_endpoint_returns_available_analysis(self, client):
        from scholar_lens.api.document_analysis import save_document_analysis

        store = documents._store
        doc = store.create_document("paper.pdf")
        store.update_status(doc.doc_id, DocumentStatus.ready)
        store.save_sections(doc.doc_id, [
            SectionSummary(section_id="intro", title="Introduction", level=1, gist="RAG setup."),
        ])
        store.save_chunks(doc.doc_id, [
            Chunk(chunk_id="c1", text="RAG improves grounding.", metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id)),
        ])
        save_document_analysis(store, doc.doc_id, parsed_doc_type="research_paper")

        r = client.get(f"/api/documents/{doc.doc_id}/analysis")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "available"
        assert data["source"] == "parser"
        assert data["key_terms"]

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
        r = client.post("/api/documents/upload/paper", files={"file": ("test.pdf", pdf, "application/pdf")})

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"

        chunks = documents._store.load_chunks(data["doc_id"])
        assert [c["metadata"]["section_id"] for c in chunks] == ["intro", "intro", "method"]

    def test_upload_falls_back_when_toc_titles_are_not_in_text(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                from scholar_lens.parsers.models import ParsedPage
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="research_paper",
                    pages=[ParsedPage(page_num=0, text="Body text without matching heading.", char_count=35)],
                    sections=[
                        {"id": "toc-intro", "title": "Introduction From Bookmark", "level": 1, "page_start": 1},
                    ],
                    raw_text="Body text without matching heading.",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def _chunk_text(self, text, doc_id, section_id, section_type, chapter):
                return []

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

        r = client.post(
            "/api/documents/upload/paper",
            files={"file": ("fallback.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        assert data["num_chunks"] == 1
        chunks = documents._store.load_chunks(data["doc_id"])
        assert chunks[0]["text"] == "Body text without matching heading."

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
            "/api/documents/upload/courseware",
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

    def test_upload_records_ocr_recommended_pages(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                from scholar_lens.parsers.models import ParsedPage
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="slides_pdf",
                    pages=[ParsedPage(page_num=5, text="", char_count=0)],
                    images=[{"page": 5, "bbox": [0, 0, 400, 300]}],
                    raw_text="",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return []

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        r = client.post(
            "/api/documents/upload/courseware",
            files={"file": ("scan.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["ocr_needed"] is True
        assert data["ocr_recommended_pages"] == [5]
        assert data["ocr_recommendation_reasons"]["5"] == "text_low_parser_visuals"

    def test_upload_does_not_mark_ocr_needed_for_weak_parseable_courseware(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                from scholar_lens.parsers.models import ParsedPage

                slide_text = "Self-attention " * 10
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="slides_pdf",
                    pages=[
                        ParsedPage(page_num=0, text=slide_text, char_count=len(slide_text)),
                        ParsedPage(page_num=1, text=slide_text, char_count=len(slide_text)),
                        ParsedPage(page_num=2, text=slide_text, char_count=len(slide_text)),
                    ],
                    sections=[],
                    raw_text="\n".join([slide_text, slide_text, slide_text]),
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return []

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        r = client.post(
            "/api/documents/upload/courseware",
            files={"file": ("parseable-slides.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["text_quality"] == "weak"
        assert data["ocr_needed"] is False
        assert data["ocr_recommended_pages"] == []

    def test_upload_saves_parse_quality_records(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                from scholar_lens.parsers.models import ParsedPage
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="slides_pdf",
                    pages=[ParsedPage(page_num=2, text="", char_count=0)],
                    images=[{"page": 2}],
                    raw_text="",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return []

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        r = client.post(
            "/api/documents/upload/courseware",
            files={"file": ("scan.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert r.status_code == 200
        data = r.json()
        qualities = documents._store.load_parse_quality(data["doc_id"])
        assert qualities[0]["unit_id"] == "page_2"
        assert qualities[0]["recommended_action"] == "ocr"
        assert data["ocr_recommended_pages"] == [2]

    def test_upload_auto_runs_ocr_and_apply_for_recommended_pages(self, client, monkeypatch):
        from scholar_lens.api.schemas import OCREnhanceResponse
        from scholar_lens.parsers.models import ParsedPage

        monkeypatch.setattr(
            documents,
            "get_settings",
            lambda: Settings(
                _env_file="",
                api_key="",
                auto_ocr_enabled=True,
                llm_quality_enabled=False,
                vision_enhancement_enabled=False,
            ),
        )

        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="slides_pdf",
                    pages=[ParsedPage(page_num=2, text="", char_count=0)],
                    images=[{"page": 2}],
                    raw_text="",
                )

        async def fake_ocr(store_arg, doc_id, mode="auto"):
            return OCREnhanceResponse(
                doc_id=doc_id,
                status="completed",
                pages=[{"page": 2, "text": "OCR text with enough slide content for chunking.", "ocr_quality": "good"}],
            )

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr(documents, "run_rapidocr_enhancement", fake_ocr)

        r = client.post(
            "/api/documents/upload/courseware",
            files={"file": ("scan.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        saved_ocr = documents._store.load_ocr_enhancement(data["doc_id"])
        assert saved_ocr["status"] == "completed"
        enhanced = documents._store.load_parsed_document(data["doc_id"], enhanced=True)
        assert "OCR text" in enhanced.raw_text
        chunks = documents._store.load_chunks(data["doc_id"])
        assert chunks
        assert "OCR text" in chunks[0]["text"]

    def test_upload_auto_runs_enabled_vision_and_apply(self, client, monkeypatch):
        from scholar_lens.api.schemas import OCREnhanceResponse, VisionEnhanceResponse
        from scholar_lens.parsers.models import ParsedPage

        class FakeParser:
            def parse(self, source):
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="slides_pdf",
                    pages=[ParsedPage(page_num=3, text="", char_count=0)],
                    images=[{"page": 3}],
                    raw_text="",
                )

        class FakeSettings:
            auto_ocr_enabled = True
            llm_quality_enabled = False
            vision_enhancement_enabled = True
            vision_api_key = "vision-key"
            vision_base_url = "https://vision.example/v1"
            vision_model = "vision-model"
            embedding = None

        async def fake_ocr(store_arg, doc_id, mode="auto"):
            return OCREnhanceResponse(
                doc_id=doc_id,
                status="failed",
                pages=[{"page": 3, "text": "", "ocr_quality": "failed", "vision_recommended": True}],
                vision_recommended_pages=[3],
            )

        async def fake_vision(store_arg, doc_id, pages, settings):
            assert pages == [3]
            assert settings is FakeSettings
            return VisionEnhanceResponse(
                doc_id=doc_id,
                status="completed",
                pages=[{"page": 3, "text": "Vision explanation for diagram content.", "vision_quality": "good"}],
            )

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr(documents, "get_settings", lambda: FakeSettings)
        monkeypatch.setattr(documents, "run_rapidocr_enhancement", fake_ocr)
        monkeypatch.setattr(documents, "run_vision_enhancement", fake_vision)

        r = client.post(
            "/api/documents/upload/courseware",
            files={"file": ("diagram.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert r.status_code == 200
        data = r.json()
        saved_vision = documents._store.load_vision_enhancement(data["doc_id"])
        assert saved_vision["status"] == "completed"
        enhanced = documents._store.load_parsed_document(data["doc_id"], enhanced=True)
        assert "Vision explanation" in enhanced.raw_text

    def test_upload_saves_parsed_document_artifact(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                from scholar_lens.parsers.models import ParsedPage
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="research_paper",
                    pages=[ParsedPage(page_num=0, text="Parsed text", char_count=11)],
                    raw_text="Parsed text",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return []

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        r = client.post(
            "/api/documents/upload/paper",
            files={"file": ("paper.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert r.status_code == 200
        parsed = documents._store.load_parsed_document(r.json()["doc_id"])
        assert parsed.raw_text == "Parsed text"

    def test_upload_does_not_recommend_sparse_nonvisual_slide(self, client, monkeypatch):
        class FakeParser:
            def parse(self, source):
                from scholar_lens.parsers.models import ParsedPage
                return ParsedDocument(
                    source_path=str(source),
                    parser_used="fake",
                    doc_subtype="slides_pdf",
                    pages=[ParsedPage(page_num=0, text="Agenda", char_count=6)],
                    raw_text="Agenda",
                )

        class FakeChunker:
            def __init__(self, max_chunk_tokens=800):
                pass

            def chunk(self, parsed, doc_id=""):
                return []

        monkeypatch.setattr("scholar_lens.parsers.pdf_parser.PDFParser", FakeParser)
        monkeypatch.setattr("scholar_lens.parsers.chunker.SectionAwareChunker", FakeChunker)

        r = client.post(
            "/api/documents/upload/courseware",
            files={"file": ("agenda.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["ocr_recommended_pages"] == []
        assert data["ocr_recommendation_reasons"] == {}

    def test_apply_enhancement_skips_when_payload_has_no_usable_text(self, client):
        from scholar_lens.api.schemas import OCREnhanceResponse
        from scholar_lens.parsers.models import ParsedPage

        store = documents._store
        doc = store.create_document("failed-ocr.pdf")
        store.save_parsed_document(doc.doc_id, ParsedDocument(
            source_path=str(store.source_path(doc.doc_id)),
            parser_used="fake",
            doc_subtype="slides_pdf",
            pages=[ParsedPage(page_num=0, text="", char_count=0)],
            raw_text="",
        ))
        store.save_ocr_enhancement(doc.doc_id, OCREnhanceResponse(
            doc_id=doc.doc_id,
            status="failed",
            pages=[{"page": 0, "text": "", "ocr_quality": "failed", "vision_recommended": True}],
            vision_recommended_pages=[0],
        ))

        r = client.post(f"/api/documents/{doc.doc_id}/enhance/apply")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "skipped"
        assert data["num_pages_updated"] == 0
