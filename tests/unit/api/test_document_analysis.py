from scholar_lens.api.document_analysis import (
    build_analysis_response,
    build_understanding_from_store_data,
    enhance_document_analysis,
    load_document_analysis,
    save_document_analysis,
)
from scholar_lens.api.schemas import SectionSummary
from scholar_lens.core.models import DocumentUnderstanding, Section, Term
from scholar_lens.core.settings import EmbeddingConfig, LLMConfig, Settings
from scholar_lens.parsers.models import Chunk, ChunkMetadata, ParsedDocument, ParsedPage
from scholar_lens.rag.document_store import DocumentStore


def test_build_understanding_from_sections_and_chunks_extracts_terms():
    sections = [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="RAG motivation."),
        SectionSummary(section_id="method", title="Method", level=1),
    ]
    chunks = [
        {
            "chunk_id": "c1",
            "text": "RAG combines retrieval with LLM generation for grounded answers.",
            "metadata": {"section_id": "intro", "doc_id": "doc1"},
        },
        {
            "chunk_id": "c2",
            "text": "The method uses self-attention and a reranker.",
            "metadata": {"section_id": "method", "doc_id": "doc1"},
        },
    ]

    understanding = build_understanding_from_store_data(
        doc_id="doc1",
        doc_type="research_paper",
        sections=sections,
        chunks=chunks,
    )

    assert understanding.l0_summaries["intro"]
    assert any(term.english == "RAG" for term in understanding.key_terms)
    assert "graph TD" in understanding.mermaid_map


def test_save_document_analysis_persists_and_loads(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="RAG problem."),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="c1",
            text="RAG improves study brief grounding.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
    ])

    saved = save_document_analysis(store, doc.doc_id, parsed_doc_type="research_paper")
    loaded = load_document_analysis(store, doc.doc_id)

    assert loaded is not None
    assert loaded.doc_type == "research_paper"
    assert saved.l0_summaries
    assert loaded.l0_summaries == saved.l0_summaries


def test_build_analysis_response_reports_missing_and_available(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="RAG problem."),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="c1",
            text="RAG improves study brief grounding.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
    ])

    missing = build_analysis_response(store, doc.doc_id)
    assert missing.status == "missing"
    assert missing.source == "missing"

    save_document_analysis(store, doc.doc_id, parsed_doc_type="research_paper")
    available = build_analysis_response(store, doc.doc_id)

    assert available.status == "available"
    assert available.source == "parser"
    assert available.key_terms
    assert available.l0_summaries["intro"]


def test_build_analysis_response_uses_parse_quality_before_llm_analysis(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    store.update_summary(
        doc.doc_id,
        ocr_recommended_pages=[2],
        ocr_recommendation_reasons={"2": "text_low_parser_visuals"},
    )
    store.save_parse_quality(doc.doc_id, [
        {
            "unit_id": "page_2",
            "unit_type": "slide",
            "page_start": 2,
            "page_end": 2,
            "text_score": 0.1,
            "structure_score": 0.4,
            "visual_score": 0.8,
            "overall_score": 0.2,
            "quality": "weak",
            "recommended_action": "ocr",
            "reasons": ["text_low", "visual_high"],
            "text_preview": "",
        }
    ])

    analysis = build_analysis_response(store, doc.doc_id)

    assert analysis.status == "available"
    assert analysis.source == "parse_quality"
    assert analysis.parse_quality_status == "needs_enhancement"
    assert analysis.parse_quality_pages[0]["page_label"] == "第 3 页"
    assert "当前配置" in analysis.parse_quality_actions[0]
    assert "解析增强" in analysis.parse_quality_actions[0]


def test_build_analysis_response_marks_completed_enhancement_as_overriding_low_quality(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    store.update_summary(
        doc.doc_id,
        ocr_recommended_pages=[2],
        ocr_recommendation_reasons={"2": "text_low_parser_visuals"},
    )
    store.save_parse_quality(doc.doc_id, [
        {
            "unit_id": "page_2",
            "unit_type": "slide",
            "page_start": 2,
            "page_end": 2,
            "text_score": 0.1,
            "structure_score": 0.4,
            "visual_score": 0.8,
            "overall_score": 0.2,
            "quality": "weak",
            "recommended_action": "ocr",
            "reasons": ["text_low", "visual_high"],
            "text_preview": "",
        }
    ])
    store.save_parsed_document(
        doc.doc_id,
        ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[ParsedPage(page_num=2, text="OCR text", char_count=8, enhanced=True)],
        ),
        enhanced=True,
    )

    analysis = build_analysis_response(store, doc.doc_id)

    assert analysis.parse_quality_status == "enhanced_completed"
    assert analysis.parse_quality_pages == []
    assert analysis.parse_quality_message == "解析增强已完成。"


async def test_enhance_document_analysis_uses_analyzer_and_persists(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="fallback"),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="c1",
            text="RAG improves study brief grounding.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
    ])

    class FakeAnalyzer:
        async def analyze_document(self, doc_text, sections, memory_manager=None, max_concurrent=5):
            assert "RAG improves" in doc_text
            assert sections[0]["section_id"] == "intro"
            return DocumentUnderstanding(
                doc_type="research_paper",
                language="en",
                difficulty="advanced",
                estimated_reading_time=8,
                sections=[Section(section_id="intro", title="Introduction", level=1)],
                mermaid_map="graph TD\n  doc-->intro",
                key_terms=[Term(english="RAG", chinese="检索增强生成")],
                l0_summaries={"intro": "LLM summary"},
                l1_overviews={"intro": "LLM overview"},
            )

    result = await enhance_document_analysis(store, doc.doc_id, analyzer=FakeAnalyzer())

    assert result.source == "llm"
    assert result.status == "enhanced"
    assert store.load_understanding(doc.doc_id).l0_summaries["intro"] == "LLM summary"


async def test_document_analysis_graph_persists_and_hydrates_memory(tmp_path):
    from scholar_lens.api.document_analysis_graph import run_document_analysis_graph
    from scholar_lens.memory.memory_manager import MemoryManager

    store = DocumentStore(root=tmp_path / "docs")
    memory = MemoryManager(data_dir=str(tmp_path / "memory"))
    doc = store.create_document("paper.pdf")
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="RAG setup."),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="c1",
            text="RAG improves grounding.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
    ])

    class FakeAnalyzer:
        async def analyze_document(self, doc_text, sections, memory_manager=None, max_concurrent=5):
            return DocumentUnderstanding(
                doc_type="research_paper",
                language="en",
                difficulty="intermediate",
                estimated_reading_time=8,
                sections=[],
                key_terms=[Term(english="RAG", chinese="检索增强生成")],
                l0_summaries={"intro": "LLM summary"},
                l1_overviews={"intro": "LLM overview"},
                mermaid_map="graph TD\n  doc-->intro",
            )

    result = await run_document_analysis_graph(
        store=store,
        doc_id=doc.doc_id,
        analyzer=FakeAnalyzer(),
        memory_manager=memory,
    )

    assert result.status == "enhanced"
    assert store.load_understanding(doc.doc_id).l0_summaries["intro"] == "LLM summary"
    assert "RAG|||检索增强生成" in memory.core_memory.active_glossary
    assert memory.document.get_l0_summary("intro") == "LLM summary"


async def test_enhance_document_analysis_without_llm_returns_unavailable(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_sections(doc.doc_id, [
        SectionSummary(section_id="intro", title="Introduction", level=1, gist="fallback"),
    ])
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="c1",
            text="RAG improves study brief grounding.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
    ])
    settings = Settings(_env_file="", llm=LLMConfig(), embedding=EmbeddingConfig())

    result = await enhance_document_analysis(store, doc.doc_id, settings=settings)

    assert result.source == "unavailable"
    assert result.status == "unavailable"
    assert result.error
    assert store.load_understanding(doc.doc_id) is None
