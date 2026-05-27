"""Shared pytest fixtures for ScholarLens tests.

Usage in tests:
    def test_foo(settings):        # clean Settings (no .env)
    def test_bar(mock_llm):        # AsyncMock LLM
    def test_baz(parser):          # PDFParser()
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path


@pytest.fixture
def clean_settings():
    """Settings with no .env file loaded — clean state for unit tests."""
    from scholar_lens.core.settings import Settings, LLMConfig, EmbeddingConfig
    return Settings(
        _env_file="",
        api_key="test-key",
        base_url="https://test.api/v1",
        llm=LLMConfig(model="test-model"),
        embedding=EmbeddingConfig(model="test-emb"),
    )


@pytest.fixture
def mock_llm():
    """AsyncMock LLM that returns empty JSON by default."""
    llm = AsyncMock()
    response = MagicMock()
    response.content = "{}"
    llm.ainvoke.return_value = response
    return llm


@pytest.fixture
def mock_llm_json():
    """AsyncMock LLM that returns valid JSON in markdown code block."""
    llm = AsyncMock()
    response = MagicMock()
    response.content = '```json\n{"doc_type":"research_paper","sections":[],"mermaid_map":""}\n```'
    llm.ainvoke.return_value = response
    return llm


@pytest.fixture
def pdf_parser():
    """PDFParser using local text extraction."""
    from scholar_lens.parsers.pdf_parser import PDFParser
    return PDFParser()


@pytest.fixture
def chunker():
    from scholar_lens.parsers.chunker import SectionAwareChunker
    return SectionAwareChunker(max_chunk_tokens=800)


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory that auto-cleans."""
    return tmp_path


@pytest.fixture
def sample_parsed_doc():
    """Minimal ParsedDocument for chunker/retriever tests."""
    from scholar_lens.parsers.models import ParsedDocument, ParsedPage
    return ParsedDocument(
        source_path="test.pdf",
        doc_subtype="research_paper",
        pages=[ParsedPage(page_num=0, text="Test content " * 20, char_count=260)],
        sections=[
            {"id": "1", "title": "Introduction", "level": 1, "text": "Introduction text " * 30},
            {"id": "2", "title": "Method", "level": 1, "text": "Method text " * 50},
            {"id": "ref", "title": "References", "level": 1, "text": "[1] Smith. 2020."},
        ],
        raw_text="Test document content.",
    )
