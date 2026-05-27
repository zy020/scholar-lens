import pytest
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip("fastapi")

from scholar_lens.api.brief_builder import build_llm_brief_prompt
from scholar_lens.api.routes.notes import build_concept_map_markdown
from scholar_lens.api.schemas import SectionSummary


def test_concept_map_markdown_uses_actual_sections():
    sections = [
        SectionSummary(section_id="intro", title="Introduction", level=1),
        SectionSummary(section_id="method", title="Method", level=1),
        SectionSummary(section_id="detail", title="Attention Layer", level=2),
    ]

    markdown = build_concept_map_markdown("paper1", sections)

    assert "# Concept Map — paper1" in markdown
    assert "Introduction" in markdown
    assert "Method" in markdown
    assert "Attention Layer" in markdown
    assert "Section 1" not in markdown
    assert "Update with actual concept relationships" not in markdown


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


def test_brief_endpoint_defaults_to_not_generated_for_ready_doc(brief_client):
    client, store = brief_client
    doc = store.create_document("paper.pdf")
    store.update_summary(doc.doc_id, text_quality="good", doc_type="research_paper")
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
    assert data["source"] == "not_generated"
    assert data["tldr"] == []
    assert data["error"] == ""
    assert "尚未生成" in data["problem"]
    assert len(data["review_questions"]) == 0


def test_notes_use_document_analysis_terms_and_map(brief_client):
    client, store = brief_client
    doc = store.create_document("paper.pdf")
    store.update_status(doc.doc_id, DocumentStatus.ready)
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="RAG setup."),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c1", text="RAG improves grounding.", metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id)),
    ])
    from scholar_lens.api.document_analysis import save_document_analysis

    save_document_analysis(store, doc.doc_id, parsed_doc_type="research_paper")

    r = client.get(f"/api/notes/{doc.doc_id}")

    assert r.status_code == 200
    data = r.json()
    assert data["concept_map"].startswith("graph TD")
    assert any(term["english"] == "RAG" for term in data["terms"])


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
    store.update_summary(doc.doc_id, text_quality="good", doc_type="research_paper")
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


async def test_brief_generation_graph_generates_and_decorates_brief():
    from scholar_lens.api.brief_graph import run_brief_generation_graph

    class FakeResponse:
        content = """
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

    class FakeLLM:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)
            return FakeResponse()

    llm = FakeLLM()
    brief = await run_brief_generation_graph(
        doc_id="doc1",
        title="paper.pdf",
        doc_type="research_paper",
        text_quality="good",
        ocr_needed=False,
        sections=[SectionSummary(section_id="method", title="Method", level=1, gist="Sparse attention.")],
        chunks=[{"text": "The paper proposes sparse self-attention.", "metadata": {"section_id": "method"}}],
        llm=llm,
    )

    assert brief.source == "llm"
    assert brief.brief_type == "paper"
    assert brief.text_quality == "good"
    assert brief.contributions[0].claim == "提出稀疏 attention block"
    assert len(llm.calls) == 1


def test_parse_lecture_llm_brief_json_preserves_core_concepts():
    raw = """
    {
      "brief_type": "lecture",
      "tldr": ["本讲介绍 backpropagation。", "核心是 chain rule。", "重点是梯度计算。"],
      "problem": "理解神经网络如何计算梯度。",
      "motivation": "反向传播是训练神经网络的基础。",
      "core_concepts": [{"claim": "Chain rule", "why_it_matters": "用于逐层计算梯度"}],
      "method_walkthrough": [{"title": "Backward pass", "explanation": "从损失函数反向传播梯度。"}],
      "key_terms": [{"term": "backpropagation", "explanation_zh": "反向传播"}],
      "reading_focus": [{"section_id": "1", "section_title": "Backpropagation", "reason": "理解核心算法"}],
      "review_questions": [{"question": "chain rule 的作用是什么？", "level": "basic"}],
      "limitations": []
    }
    """

    brief = parse_llm_brief_json("doc1", "lecture.pdf", raw)

    assert brief.brief_type == "lecture"
    assert brief.contributions[0].claim == "Chain rule"


from scholar_lens.api.brief_builder import build_lecture_llm_brief_prompt


def test_lecture_llm_prompt_is_not_paper_template():
    prompt = build_lecture_llm_brief_prompt(
        "lecture.pdf",
        [SectionSummary(section_id="1", title="Backpropagation", level=1, gist="Gradient descent and chain rule.")],
        [{"text": "Backpropagation applies the chain rule to neural networks.", "metadata": {"section_id": "1"}}],
    )

    assert "Lecture Study Brief" in prompt
    assert "core_concepts" in prompt
    assert "important_slides" in prompt
    assert "formulas_or_figures" in prompt
    assert "self-check" in prompt
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


def test_brief_endpoint_does_not_turn_low_text_quality_into_brief(brief_client):
    client, store = brief_client
    doc = store.create_document("scanned-slides.pdf")
    store.update_summary(
        doc.doc_id,
        doc_type="slides_pdf",
        text_quality="image_based",
        ocr_needed=True,
        page_text_coverage=0.0,
        section_quality="none",
        diagnostic_notes=["当前 PDF 疑似图片型课件，文本抽取不足，建议启用 OCR 或 Vision Model。"],
    )
    store.update_status(doc.doc_id, DocumentStatus.ready)
    store.save_sections(doc.doc_id, [])
    store.save_chunks(doc.doc_id, [])

    r = client.get(f"/api/notes/{doc.doc_id}/brief?force=true")

    assert r.status_code == 200
    data = r.json()
    assert data["brief_type"] == "lecture"
    assert data["text_quality"] == "image_based"
    assert data["ocr_needed"] is True
    assert data["source"] == "unavailable"
    assert data["tldr"] == []


def test_brief_endpoint_requires_llm_for_text_slides(brief_client):
    client, store = brief_client
    doc = store.create_document("lecture.pdf")
    store.update_summary(
        doc.doc_id,
        doc_type="slides_pdf",
        text_quality="good",
        ocr_needed=False,
        page_text_coverage=1.0,
        section_quality="good",
    )
    store.update_status(doc.doc_id, DocumentStatus.ready)
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="l1", title="Lecture 1: Attention", level=1, gist="Attention connects tokens."),
        SectionSummary(section_id="l2", title="Example", level=1, gist="Translation example."),
        SectionSummary(section_id="l3", title="Exercise", level=1, gist="Practice question."),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c1", text="Self-attention computes token relationships.", metadata=ChunkMetadata(section_id="l1", doc_id=doc.doc_id)),
    ])

    r = client.get(f"/api/notes/{doc.doc_id}/brief?force=true")

    assert r.status_code == 200
    data = r.json()
    assert data["brief_type"] == "lecture"
    assert data["text_quality"] == "good"
    assert data["source"] == "unavailable"
    assert data["tldr"] == []


def test_brief_endpoint_keeps_legacy_unknown_text_quality_as_paper_brief_type(brief_client):
    client, store = brief_client
    doc = store.create_document("legacy-paper.pdf")
    store.update_summary(doc.doc_id, doc_type="research_paper")
    store.update_status(doc.doc_id, DocumentStatus.ready)
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="Problem setup."),
        SectionSummary(section_id="method", title="Method", level=1, gist="Method setup."),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c1", text="Legacy paper text about Transformer methods.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
    ])

    r = client.get(f"/api/notes/{doc.doc_id}/brief?force=true")

    assert r.status_code == 200
    data = r.json()
    assert data["brief_type"] == "paper"
    assert data["brief_type"] != "low_text"
    assert data["source"] == "unavailable"


def test_slides_pdf_routes_to_lecture_brief(brief_client):
    client, store = brief_client
    doc = store.create_document("week-1-course.pdf")
    store.update_summary(doc.doc_id, doc_type="slides_pdf", text_quality="good")
    store.update_status(doc.doc_id, DocumentStatus.ready)
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="s1", title="Week 1: Neural Networks", level=1, gist="Course overview."),
        SectionSummary(section_id="s2", title="Lecture Example", level=1, gist="Worked example."),
        SectionSummary(section_id="s3", title="Exercise", level=1, gist="Practice."),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c1", text="This lecture explains neural network basics.", metadata=ChunkMetadata(section_id="s1", doc_id=doc.doc_id)),
    ])

    r = client.get(f"/api/notes/{doc.doc_id}/brief?force=true")

    assert r.status_code == 200
    assert r.json()["brief_type"] == "lecture"
    assert r.json()["source"] == "unavailable"


def test_brief_endpoint_does_not_fallback_when_llm_fails(tmp_path, monkeypatch):
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

    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = RuntimeError("model timeout")

    class FakeFactory:
        def create(self, config=None, streaming=True):
            return mock_llm

    from scholar_lens.core import llm_factory
    monkeypatch.setattr(llm_factory.ChatLLMFactory, "from_settings", lambda settings: FakeFactory())

    doc = store.create_document("paper.pdf")
    store.update_summary(doc.doc_id, text_quality="good", doc_type="research_paper")
    store.update_status(doc.doc_id, DocumentStatus.ready)
    store.save_sections(doc.doc_id, [SectionSummary(section_id="intro", title="Introduction", level=1, gist="Problem setup.")])
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c1", text="The paper studies a RAG-based study brief.", metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id)),
    ])
    app = create_app()
    client = ASGITestClient(app)

    r = client.get(f"/api/notes/{doc.doc_id}/brief?force=true")

    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "unavailable"
    assert data["tldr"] == []
    assert "model timeout" in data["error"]
    assert not (store.document_dir(doc.doc_id) / "paper_brief.json").exists()
