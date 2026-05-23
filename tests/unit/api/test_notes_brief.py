"""Tests for paper brief builder and endpoint."""
import pytest

pytest.importorskip("fastapi")

from scholar_lens.api.brief_builder import build_fallback_brief
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
from scholar_lens.api.routes import documents, notes
from scholar_lens.api.schemas import DocumentStatus
from scholar_lens.parsers.models import Chunk, ChunkMetadata
from scholar_lens.rag.document_store import DocumentStore
from tests.unit.api.helpers import ASGITestClient


@pytest.fixture
def brief_client(tmp_path, monkeypatch):
    store = DocumentStore(root=tmp_path)
    monkeypatch.setattr(documents, "_store", store)
    monkeypatch.setattr(notes, "_store", store)
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
