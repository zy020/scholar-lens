import pytest
from scholar_lens.api.schemas import (
    ConfigUpdateRequest,
    ConfigResponse,
    DocumentUploadResponse,
    ChatRequest,
    ChatMessage,
    ExplanationResponse,
    NotesResponse,
)


class TestSchemas:
    def test_config_update_request(self):
        req = ConfigUpdateRequest(
            llm_api_key="key",
            llm_model="gpt-4o-mini",
            embedding_api_key="emb-key",
        )
        assert req.llm_model == "gpt-4o-mini"

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
        )
        assert req.message == "Explain self-attention"

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"

    def test_document_upload_response(self):
        resp = DocumentUploadResponse(
            doc_id="paper_001",
            doc_type="research_paper",
            num_sections=5,
            status="processed",
        )
        assert resp.status == "processed"

    def test_explanation_response(self):
        resp = ExplanationResponse(
            original="Self-attention",
            translation="自注意力",
            explanation="A mechanism...",
            confidence="high",
        )
        assert resp.confidence == "high"
