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


def test_fallback_brief_preserves_english_terms():
    sections = [SectionSummary(section_id="m", title="Method", level=1, gist="Transformer attention.")]
    chunks = [
        {"text": "Transformer and self-attention are central to the proposed model.", "metadata": {"section_id": "m"}},
    ]

    brief = build_fallback_brief("doc1", "paper.pdf", sections, chunks)

    terms = {term.term for term in brief.key_terms}
    assert "Transformer" in terms or "self-attention" in terms
