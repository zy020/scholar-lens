"""Test DocumentIndex per Phase 1 spec: overlap fallback without BM25, evidence format."""
import tempfile, shutil
from pathlib import Path
from scholar_lens.rag.document_store import DocumentStore
from scholar_lens.rag.document_index import DocumentIndex, evidence_from_results
from scholar_lens.api.schemas import DocumentStatus
from scholar_lens.parsers.models import Chunk, ChunkMetadata


class TestDocumentIndex:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.store = DocumentStore(root=self.tmp)
        self.idx = DocumentIndex(self.store)

    def _seed(self, doc_id, texts):
        chunks = [Chunk(chunk_id=f"c{i}", text=t, metadata=ChunkMetadata(section_id=str(i%3), doc_id=doc_id), layer="L2") for i, t in enumerate(texts)]
        self.store.save_chunks(doc_id, chunks)

    def test_overlap_fallback_without_bm25(self):
        # rank_bm25 may or may not be installed; search must work either way
        doc = self.store.create_document("test.pdf")
        self._seed(doc.doc_id, ["Self-attention relates positions", "Adam optimizer training", "Transformer architecture"])
        self.store.update_status(doc.doc_id, DocumentStatus.ready)
        self.idx._bm25_available = False
        results = self.idx.search(doc.doc_id, "self-attention", top_k=2)
        assert len(results) > 0
        assert results[0].score > 0

    def test_section_boost(self):
        doc = self.store.create_document("test.pdf")
        self._seed(doc.doc_id, ["Attention mechanism overview", "The transformer uses attention", "Unrelated text about apples"])
        self.store.update_status(doc.doc_id, DocumentStatus.ready)
        results = self.idx.search(doc.doc_id, "attention", section_id="0", top_k=3)
        assert results[0].chunk_id == "c0"  # section boost puts matching chunk first

    def test_section_only_mode(self):
        doc = self.store.create_document("test.pdf")
        self._seed(doc.doc_id, ["Self-attention in section 0", "Transformers overview in section 1"])
        self.store.update_status(doc.doc_id, DocumentStatus.ready)
        results = self.idx.search(doc.doc_id, "self-attention", section_id="0", section_only=True, top_k=3)
        assert len(results) == 1  # only section 0 chunk
        assert "section 0" in results[0].text

    def test_evidence_format(self):
        doc = self.store.create_document("test.pdf")
        self._seed(doc.doc_id, ["Self-attention mechanism"])
        self.store.update_status(doc.doc_id, DocumentStatus.ready)
        results = self.idx.search(doc.doc_id, "attention", top_k=1)
        evidence = evidence_from_results(results)
        assert len(evidence) > 0
        e = evidence[0]
        for key in ("doc_id", "chunk_id", "quote", "score"):
            assert key in e
        assert e["quote"] == "Self-attention mechanism"

    def test_empty_chunks_returns_empty(self):
        doc = self.store.create_document("empty.pdf")
        self.store.update_status(doc.doc_id, DocumentStatus.ready)
        assert self.idx.search(doc.doc_id, "query") == []

    def test_courseware_search_boosts_slide_title_core_terms_over_question_words(self):
        doc = self.store.create_document("slides.pdf")
        chunks = [
            Chunk(
                chunk_id="generic",
                text="Model\nWhat is the output?\nEach vector has a label.",
                metadata=ChunkMetadata(
                    section_id="slide_7",
                    section_type="slide",
                    content_type="slide",
                    page_start=7,
                    doc_id=doc.doc_id,
                ),
            ),
            Chunk(
                chunk_id="core",
                text="Self-attention\nAttention is all you need.",
                metadata=ChunkMetadata(
                    section_id="slide_11",
                    section_type="slide",
                    content_type="slide",
                    page_start=11,
                    doc_id=doc.doc_id,
                ),
            ),
        ]
        self.store.save_chunks(doc.doc_id, chunks)
        self.store.update_status(doc.doc_id, DocumentStatus.ready)

        results = self.idx.search(doc.doc_id, "What is self-attention?", top_k=2)

        assert results[0].chunk_id == "core"

    def test_formula_metadata_terms_are_searchable(self):
        doc = self.store.create_document("slides.pdf")
        chunks = [
            Chunk(
                chunk_id="plain",
                text="Attention overview without variables.",
                metadata=ChunkMetadata(
                    section_id="slide_5",
                    section_type="slide",
                    content_type="slide",
                    page_start=5,
                    doc_id=doc.doc_id,
                ),
            ),
            Chunk(
                chunk_id="formula",
                text="Formula\n𝛼 = softmax(𝒒∙𝒌)",
                metadata=ChunkMetadata(
                    section_id="slide_4",
                    section_type="slide",
                    content_type="slide",
                    page_start=4,
                    doc_id=doc.doc_id,
                    has_formula=True,
                    formula_ids=["q dot k", "alpha softmax q k"],
                    contextual_prefix="Formula terms: q dot k; alpha softmax q k",
                ),
            ),
        ]
        self.store.save_chunks(doc.doc_id, chunks)
        self.store.update_status(doc.doc_id, DocumentStatus.ready)
        self.idx._bm25_available = False

        results = self.idx.search(doc.doc_id, "explain q dot k alpha", top_k=2)

        assert results[0].chunk_id == "formula"
        assert results[0].score > results[1].score

    def test_unicode_formula_query_is_normalized_for_search(self):
        doc = self.store.create_document("slides.pdf")
        chunks = [
            Chunk(
                chunk_id="plain",
                text="Self-attention vs CNN compares receptive fields.",
                metadata=ChunkMetadata(
                    section_id="slide_34",
                    section_type="slide",
                    content_type="slide",
                    page_start=34,
                    doc_id=doc.doc_id,
                ),
            ),
            Chunk(
                chunk_id="formula",
                text="Formula\n𝛼 = softmax(𝒒∙𝒌)",
                metadata=ChunkMetadata(
                    section_id="slide_14",
                    section_type="slide",
                    content_type="slide",
                    page_start=14,
                    doc_id=doc.doc_id,
                    has_formula=True,
                    formula_ids=["q dot k", "alpha softmax q k"],
                    contextual_prefix="Formula terms: q dot k; alpha softmax q k",
                ),
            ),
        ]
        self.store.save_chunks(doc.doc_id, chunks)
        self.store.update_status(doc.doc_id, DocumentStatus.ready)
        self.idx._bm25_available = False

        results = self.idx.search(doc.doc_id, "解释公式 𝒒∙𝒌 在 self-attention 里的含义", top_k=2)

        assert results[0].chunk_id == "formula"

    def test_latex_formula_query_is_normalized_for_search(self):
        doc = self.store.create_document("slides.pdf")
        chunks = [
            Chunk(
                chunk_id="plain",
                text="Self-attention overview without the formula.",
                metadata=ChunkMetadata(
                    section_id="slide_10",
                    section_type="slide",
                    content_type="slide",
                    page_start=10,
                    doc_id=doc.doc_id,
                ),
            ),
            Chunk(
                chunk_id="formula",
                text="Formula\n𝛼 = softmax(𝒒∙𝒌)",
                metadata=ChunkMetadata(
                    section_id="slide_14",
                    section_type="slide",
                    content_type="slide",
                    page_start=14,
                    doc_id=doc.doc_id,
                    has_formula=True,
                    formula_ids=["q dot k", "alpha softmax q k"],
                    contextual_prefix="Formula terms: q dot k; alpha softmax q k",
                ),
            ),
        ]
        self.store.save_chunks(doc.doc_id, chunks)
        self.store.update_status(doc.doc_id, DocumentStatus.ready)
        self.idx._bm25_available = False

        results = self.idx.search(doc.doc_id, r"explain \\alpha = softmax(q \\cdot k)", top_k=2)

        assert results[0].chunk_id == "formula"
