"""Tests for paper brief builder and endpoint."""
import pytest
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip("fastapi")

from scholar_lens.api.brief_builder import build_fallback_brief, build_llm_brief_prompt
from scholar_lens.api.schemas import SectionSummary


def test_fallback_brief_has_useful_sections():
    sections = [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="This paper studies efficient attention for long documents."),
        SectionSummary(section_id="method", title="Model Architecture", level=1, gist="The method introduces sparse attention blocks."),
        SectionSummary(section_id="exp", title="Experiments", level=1, gist="Experiments compare accuracy and speed."),
    ]
    chunks = [
        {"text": "The paper addresses the quadratic cost of self-attention in long document modeling.", "metadata": {"section_id": "intro"}},
        {"text": "The model uses sparse attention blocks to reduce computation while preserving context.", "metadata": {"section_id": "method"}},
        {"text": "Experiments show improved throughput with comparable accuracy.", "metadata": {"section_id": "exp"}},
    ]

    brief = build_fallback_brief("doc1", "paper.pdf", sections, chunks)

    assert brief.doc_id == "doc1"
    assert brief.source == "fallback"
    assert len(brief.tldr) >= 3
    assert brief.problem
    assert len(brief.contributions) >= 2
    assert len(brief.method_walkthrough) >= 2
    assert len(brief.reading_focus) >= 3
    assert len(brief.review_questions) >= 5
    assert brief.contributions[0].evidence is not None


from scholar_lens.api.main import create_app
from scholar_lens.api import deps
from scholar_lens.api.routes import documents, notes
from scholar_lens.api.schemas import DocumentStatus
from scholar_lens.core.settings import EmbeddingConfig, LLMConfig, Settings
from scholar_lens.parsers.models import Chunk, ChunkMetadata
from scholar_lens.rag.document_store import DocumentStore
from tests.unit.api.helpers import ASGITestClient


@pytest.fixture
def brief_client(tmp_path, monkeypatch):
    store = DocumentStore(root=tmp_path)
    monkeypatch.setattr(documents, "_store", store)
    monkeypatch.setattr(notes, "_store", store)
    monkeypatch.setattr(
        deps,
        "get_settings",
        lambda: Settings(
            _env_file="",
            api_key="",
            llm=LLMConfig(),
            embedding=EmbeddingConfig(),
        ),
    )
    app = create_app()
    return ASGITestClient(app), store


def test_brief_endpoint_returns_fallback_for_ready_doc(brief_client):
    client, store = brief_client
    doc = store.create_document("paper.pdf")
    store.update_status(doc.doc_id, DocumentStatus.ready)
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="Problem setup."),
        SectionSummary(section_id="method", title="Method", level=1, gist="Method setup."),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c1", text="Transformer self-attention method.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
    ])

    r = client.get(f"/api/notes/{doc.doc_id}/brief")

    assert r.status_code == 200
    data = r.json()
    assert data["doc_id"] == doc.doc_id
    assert data["source"] == "fallback"
    assert len(data["tldr"]) >= 3
    assert len(data["review_questions"]) >= 5


def test_brief_endpoint_404_for_missing_doc(brief_client):
    client, _store = brief_client

    r = client.get("/api/notes/missing/brief")

    assert r.status_code == 404


def test_brief_endpoint_uses_brief_specific_timeout(tmp_path, monkeypatch):
    store = DocumentStore(root=tmp_path)
    monkeypatch.setattr(documents, "_store", store)
    monkeypatch.setattr(notes, "_store", store)
    monkeypatch.setattr(
        deps,
        "get_settings",
        lambda: Settings(
            _env_file="",
            api_key="test-key",
            llm=LLMConfig(api_key="test-key", model="test-model", request_timeout=60),
            embedding=EmbeddingConfig(),
        ),
    )

    captured = {}
    mock_response = MagicMock()
    mock_response.content = """
    {
      "tldr": ["问题是长文本阅读困难。", "方法是生成结构化 brief。", "实验用于验证效果。"],
      "problem": "学生难以快速理解论文。",
      "motivation": "降低阅读门槛。",
      "contributions": [{"claim": "结构化总结", "why_it_matters": "便于复习"}],
      "method_walkthrough": [{"title": "Brief generation", "explanation": "抽取章节和证据。"}],
      "key_terms": [{"term": "RAG", "explanation_zh": "检索增强生成"}],
      "reading_focus": [{"section_id": "intro", "section_title": "Introduction", "reason": "理解问题"}],
      "review_questions": [{"question": "核心问题是什么？", "level": "basic"}],
      "limitations": ["需要人工核对。"]
    }
    """
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_response

    class FakeFactory:
        def create(self, config=None, streaming=True):
            captured["config"] = config
            captured["streaming"] = streaming
            return mock_llm

    from scholar_lens.core import llm_factory
    monkeypatch.setattr(llm_factory.ChatLLMFactory, "from_settings", lambda settings: FakeFactory())

    doc = store.create_document("paper.pdf")
    store.update_status(doc.doc_id, DocumentStatus.ready)
    store.save_sections(doc.doc_id, [SectionSummary(section_id="intro", title="Introduction", level=1, gist="Problem setup.")])
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c1", text="The paper studies a RAG-based study brief.", metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id)),
    ])
    app = create_app()
    client = ASGITestClient(app)

    r = client.get(f"/api/notes/{doc.doc_id}/brief?force=true")

    assert r.status_code == 200
    assert r.json()["source"] == "llm"
    assert captured["config"].request_timeout >= 120
    assert captured["streaming"] is False


from scholar_lens.api.brief_builder import parse_llm_brief_json


def test_parse_llm_brief_json_normalizes_required_fields():
    raw = """
    {
      "tldr": ["问题是长文本建模成本高。", "方法是稀疏 attention。", "实验展示吞吐提升。"],
      "problem": "长文本 self-attention 成本高。",
      "motivation": "降低计算成本。",
      "contributions": [{"claim": "提出稀疏 attention block", "why_it_matters": "降低复杂度"}],
      "method_walkthrough": [{"title": "Sparse block", "explanation": "减少注意力连接。"}],
      "key_terms": [{"term": "self-attention", "explanation_zh": "自注意力机制"}],
      "reading_focus": [{"section_id": "method", "section_title": "Method", "reason": "理解方法"}],
      "review_questions": [{"question": "方法核心是什么？", "level": "basic"}],
      "limitations": ["未验证超长文本。"]
    }
    """

    brief = parse_llm_brief_json("doc1", "paper.pdf", raw)

    assert brief.source == "llm"
    assert brief.contributions[0].claim == "提出稀疏 attention block"
    assert brief.key_terms[0].term == "self-attention"


def test_fallback_brief_preserves_english_terms():
    sections = [SectionSummary(section_id="m", title="Method", level=1, gist="Transformer attention.")]
    chunks = [
        {"text": "Transformer and self-attention are central to the proposed model.", "metadata": {"section_id": "m"}},
    ]

    brief = build_fallback_brief("doc1", "paper.pdf", sections, chunks)

    terms = {term.term for term in brief.key_terms}
    assert "Transformer" in terms or "self-attention" in terms


from scholar_lens.api.brief_builder import build_lecture_fallback_brief, build_low_text_brief, build_lecture_llm_brief_prompt


def test_lecture_fallback_brief_uses_learning_structure():
    sections = [
        SectionSummary(section_id="1", title="Lecture 1: Attention", level=1, gist="Attention mechanisms connect tokens."),
        SectionSummary(section_id="2", title="Example: Translation", level=1, gist="Examples show alignment."),
    ]
    chunks = [
        {"text": "Self-attention computes relationships between tokens in a sequence.", "metadata": {"section_id": "1"}},
        {"text": "The lecture example explains machine translation alignment.", "metadata": {"section_id": "2"}},
    ]

    brief = build_lecture_fallback_brief("doc1", "lecture.pdf", sections, chunks, text_quality="good", ocr_needed=False)

    assert brief.brief_type == "lecture"
    assert brief.text_quality == "good"
    assert brief.ocr_needed is False
    assert any("知识点" in item or "本讲" in item for item in brief.tldr)
    assert any(q.level in {"basic", "deep", "critical"} for q in brief.review_questions)


def test_low_text_brief_warns_instead_of_hallucinating():
    brief = build_low_text_brief(
        doc_id="doc1",
        title="scanned-slides.pdf",
        doc_type="slides_pdf",
        text_quality="image_based",
        diagnostic_notes=["当前 PDF 疑似图片型课件，文本抽取不足，建议启用 OCR 或 Vision Model。"],
    )

    assert brief.brief_type == "low_text"
    assert brief.ocr_needed is True
    assert brief.text_quality == "image_based"
    assert "文本抽取不足" in brief.problem
    assert "OCR" in " ".join(brief.limitations)
    assert brief.contributions == []
    assert brief.method_walkthrough == []


def test_lecture_llm_prompt_is_not_paper_template():
    prompt = build_lecture_llm_brief_prompt(
        "lecture.pdf",
        [SectionSummary(section_id="1", title="Backpropagation", level=1, gist="Gradient descent and chain rule.")],
        [{"text": "Backpropagation applies the chain rule to neural networks.", "metadata": {"section_id": "1"}}],
    )

    assert "Lecture Study Brief" in prompt
    assert "core_concepts" in prompt
    assert "Contribution Map" not in prompt
    assert len(prompt) <= 7000


def test_llm_brief_prompt_stays_compact_for_long_documents():
    sections = [
        SectionSummary(
            section_id=f"s{i}",
            title=f"Section {i}",
            level=1,
            gist="This is a deliberately long section gist about Transformer architectures. " * 20,
        )
        for i in range(30)
    ]
    chunks = [
        {
            "text": "This chunk discusses self-attention, retrieval, experiments, and limitations. " * 80,
            "metadata": {"section_id": f"s{i % 10}"},
        }
        for i in range(30)
    ]

    prompt = build_llm_brief_prompt("long-paper.pdf", sections, chunks)

    assert len(prompt) <= 7000
    assert "Return ONLY valid JSON" in prompt
