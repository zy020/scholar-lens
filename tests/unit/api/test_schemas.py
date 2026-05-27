import pytest
from scholar_lens.api.schemas import (
    ConfigUpdateRequest,
    ConfigResponse,
    DocumentStatus,
    DocumentSummary,
    DocumentDetail,
    SectionSummary,
    EvidenceItem,
    ChatRequest,
    ChatMessage,
    ExplanationResponse,
    NotesResponse,
)


class TestSchemas:
    # New schemas (Ticket 1)
    def test_document_status_enum(self):
        assert DocumentStatus.uploaded.value == "uploaded"
        assert DocumentStatus.ready.value == "ready"
        assert DocumentStatus.failed.value == "failed"

    def test_document_summary_defaults(self):
        ds = DocumentSummary(doc_id="abc", name="test.pdf")
        assert ds.status == DocumentStatus.uploaded
        assert ds.file_url == ""
        assert ds.num_sections == 0
        assert ds.num_chunks == 0

    def test_document_summary_serialization(self):
        ds = DocumentSummary(doc_id="abc", name="test.pdf", status=DocumentStatus.ready,
                             doc_type="research_paper", file_url="/api/documents/abc/file",
                             num_sections=5, num_chunks=20)
        data = ds.model_dump()
        assert data["doc_id"] == "abc"
        assert data["status"] == "ready"
        assert data["file_url"] == "/api/documents/abc/file"

    def test_document_summary_includes_ocr_recommendations(self):
        summary = DocumentSummary(
            doc_id="doc1",
            name="slides.pdf",
            ocr_recommended_pages=[2, 4],
            ocr_recommendation_reasons={
                "2": "text_low_visual_high",
                "4": "text_low_parser_visuals",
            },
        )

        data = summary.model_dump()

        assert data["ocr_recommended_pages"] == [2, 4]
        assert data["ocr_recommendation_reasons"]["2"] == "text_low_visual_high"

    def test_enhance_plan_response_schema(self):
        from scholar_lens.api.schemas import EnhancePlanResponse

        resp = EnhancePlanResponse(
            doc_id="doc1",
            status="planned",
            recommended_ocr_pages=[1, 3],
            ocr_recommendation_reasons={"1": "text_low_parser_visuals"},
            estimated_ocr_pages=2,
            vision_available=True,
            vision_enhancement_enabled=True,
            vision_possible=True,
            vision_escalation_reasons=["ocr_too_short_visual_high", "diagram_like"],
            ocr_engine="rapidocr",
            ocr_installed=True,
            ocr_gpu_available=False,
            ocr_cpu_available=False,
            ocr_recommended_mode="vision_only",
            available_actions=["vision"],
        )

        data = resp.model_dump()

        assert data["status"] == "planned"
        assert data["estimated_ocr_pages"] == 2
        assert data["vision_possible"] is True
        assert data["vision_enhancement_enabled"] is True
        assert data["ocr_engine"] == "rapidocr"
        assert data["ocr_recommended_mode"] == "vision_only"
        assert data["available_actions"] == ["vision"]

    def test_parse_quality_response_schema(self):
        from scholar_lens.api.schemas import ParseQualityResponse

        resp = ParseQualityResponse(
            doc_id="doc1",
            source="heuristic",
            status="available",
            qualities=[{"unit_id": "page_1", "quality": "good"}],
            message="ok",
        )

        data = resp.model_dump()

        assert data["doc_id"] == "doc1"
        assert data["source"] == "heuristic"
        assert data["qualities"][0]["unit_id"] == "page_1"

    def test_ocr_enhance_response_schema(self):
        from scholar_lens.api.schemas import OCREnhanceResponse

        resp = OCREnhanceResponse(
            doc_id="doc1",
            status="completed",
            engine="rapidocr",
            pages=[{"page": 2, "text": "OCR text", "ocr_quality": "good"}],
            vision_recommended_pages=[3],
            message="ok",
        )

        data = resp.model_dump()

        assert data["doc_id"] == "doc1"
        assert data["engine"] == "rapidocr"
        assert data["pages"][0]["page"] == 2
        assert data["vision_recommended_pages"] == [3]

    def test_vision_enhance_response_schema(self):
        from scholar_lens.api.schemas import VisionEnhanceResponse

        resp = VisionEnhanceResponse(
            doc_id="doc1",
            status="completed",
            engine="vision",
            pages=[{"page": 3, "text": "Vision explanation", "vision_quality": "good"}],
            message="ok",
        )

        data = resp.model_dump()

        assert data["doc_id"] == "doc1"
        assert data["engine"] == "vision"
        assert data["pages"][0]["page"] == 3

    def test_enhancement_apply_response_schema(self):
        from scholar_lens.api.schemas import EnhancementApplyResponse

        resp = EnhancementApplyResponse(
            doc_id="doc1",
            status="applied",
            source="ocr",
            num_pages_updated=2,
            num_chunks=4,
            message="ok",
        )

        data = resp.model_dump()

        assert data["status"] == "applied"
        assert data["num_pages_updated"] == 2
        assert data["num_chunks"] == 4

    def test_document_detail_inherits_summary(self):
        dd = DocumentDetail(doc_id="abc", name="test.pdf",
                            sections=[SectionSummary(section_id="1", title="Intro", gist="test")])
        assert dd.doc_id == "abc"
        assert len(dd.sections) == 1
        assert dd.sections[0].title == "Intro"

    def test_section_summary_defaults(self):
        ss = SectionSummary(section_id="3.1", title="Methods")
        assert ss.level == 1
        assert ss.page_start is None
        assert ss.gist == ""

    def test_evidence_item(self):
        ei = EvidenceItem(doc_id="abc", chunk_id="c1", quote="text", score=0.85)
        assert ei.section_id == ""
        assert ei.page is None
        assert ei.quote == "text"

    def test_config_response_requires_restart(self):
        cr = ConfigResponse(llm_model="m", embedding_model="e")
        assert cr.requires_restart is False

    # Existing schemas
    def test_config_update_request(self):
        req = ConfigUpdateRequest(
            llm_api_key="key",
            llm_model="gpt-4o-mini",
            embedding_api_key="emb-key",
            reranker_api_key="rerank-key",
            reranker_base_url="https://rerank.example/v1",
            reranker_model="rerank-model",
            vision_api_key="vision-key",
            vision_base_url="https://vision.example/v1",
            vision_model="vision-model",
        )
        assert req.llm_model == "gpt-4o-mini"
        assert req.reranker_api_key == "rerank-key"
        assert req.reranker_base_url == "https://rerank.example/v1"
        assert req.vision_base_url == "https://vision.example/v1"

    def test_config_update_model_enable_and_separate_flags(self):
        req = ConfigUpdateRequest(
            reranker_enabled=True,
            vision_enabled=True,
            llm_use_separate=True,
            embedding_use_separate=False,
            reranker_use_separate=True,
            vision_use_separate=False,
        )
        assert req.reranker_enabled is True
        assert req.vision_enabled is True
        assert req.llm_use_separate is True
        assert req.embedding_use_separate is False

    def test_config_response(self):
        resp = ConfigResponse(
            llm_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            reranker_available=False,
            vision_available=False,
        )
        assert resp.reranker_available is False

    def test_chat_request(self):
        req = ChatRequest(
            message="Explain self-attention",
            doc_id="paper_001",
            section_id="3.1",
            top_k=3,
            section_only=True,
            use_reranker=False,
            student_level="advanced",
            deep_mode=True,
        )
        assert req.message == "Explain self-attention"
        assert req.top_k == 3
        assert req.section_only is True
        assert req.use_reranker is False
        assert req.student_level == "advanced"
        assert req.deep_mode is True

    def test_chat_request_accepts_context_k(self):
        req = ChatRequest(message="Explain self-attention", top_k=2, context_k=4)

        assert req.context_k == 4

        with pytest.raises(ValueError):
            ChatRequest(message="Explain self-attention", context_k=41)

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"

    def test_document_upload_response_backward_compat(self):
        ds = DocumentSummary(doc_id="paper_001", name="test.pdf", doc_type="research_paper",
                             num_sections=5, status=DocumentStatus.ready)
        assert ds.doc_id == "paper_001"

    def test_explanation_response(self):
        resp = ExplanationResponse(
            original="Self-attention",
            translation="自注意力",
            explanation="A mechanism...",
            confidence="high",
        )
        assert resp.confidence == "high"
