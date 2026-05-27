import pytest
from scholar_lens.rag.vectorstore import ScholarVectorStore
from scholar_lens.parsers.models import Chunk, ChunkMetadata


class TestScholarVectorStore:
    def test_instantiation(self):
        vs = ScholarVectorStore(collection_name="test", persist_dir=None)
        assert vs is not None

    def test_add_and_query(self):
        vs = ScholarVectorStore(collection_name="test_add", persist_dir=None)
        chunks = [
            Chunk(chunk_id="doc1_1_0", text="Self-attention mechanism allows each position to attend to all other positions.", metadata=ChunkMetadata(section_id="3.1", section_type="method", doc_id="doc1", formula_ids=["x"], cross_refs=["x"]), layer="L2"),
            Chunk(chunk_id="doc1_2_0", text="The training procedure uses Adam optimizer with learning rate warmup.", metadata=ChunkMetadata(section_id="4", section_type="method", doc_id="doc1", formula_ids=["x"], cross_refs=["x"]), layer="L2"),
        ]
        vs.add_chunks(chunks, embeddings=[[0.1] * 10, [0.2] * 10])
        results = vs.query(query_embedding=[0.1] * 10, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "doc1_1_0"

    def test_query_empty_store(self):
        vs = ScholarVectorStore(collection_name="test_empty", persist_dir=None)
        results = vs.query(query_embedding=[0.1] * 10, top_k=5)
        assert results == []

    def test_delete_by_doc_id(self):
        vs = ScholarVectorStore(collection_name="test_delete", persist_dir=None)
        chunks = [
            Chunk(chunk_id="doc1_1_0", text="Hello", metadata=ChunkMetadata(section_id="1", section_type="prose", doc_id="doc1", formula_ids=["x"], cross_refs=["x"]), layer="L2"),
        ]
        vs.add_chunks(chunks, embeddings=[[0.1] * 10])
        vs.delete_by_doc_id("doc1")
        results = vs.query(query_embedding=[0.1] * 10, top_k=5)
        assert len(results) == 0

    def test_new_chunk_metadata_roundtrip(self):
        vs = ScholarVectorStore(collection_name="test_metadata_roundtrip", persist_dir=None)
        chunk = Chunk(
            chunk_id="doc1_slide_3_0",
            text="OCR enhanced slide text.",
            metadata=ChunkMetadata(
                section_id="slide_3",
                section_type="slide",
                doc_id="doc1",
                page_start=3,
                page_end=3,
                content_source="ocr",
                enhanced=True,
            ),
            layer="L2",
        )

        vs.add_chunks([chunk], embeddings=[[0.1] * 10])
        results = vs.query(query_embedding=[0.1] * 10, top_k=1)

        assert results[0].metadata.page_start == 3
        assert results[0].metadata.page_end == 3
        assert results[0].metadata.content_source == "ocr"
        assert results[0].metadata.enhanced is True

    def test_query_results_returns_scored_retrieval_results(self):
        vs = ScholarVectorStore(collection_name="test_query_results", persist_dir=None)
        chunk = Chunk(
            chunk_id="doc1_1_0",
            text="Vector retrieval evidence.",
            metadata=ChunkMetadata(section_id="1", section_type="prose", doc_id="doc1"),
            layer="L2",
        )

        vs.add_chunks([chunk], embeddings=[[0.1] * 10])
        results = vs.query_results(query_embedding=[0.1] * 10, top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "doc1_1_0"
        assert results[0].source == "vector"
        assert results[0].rank == 1
        assert results[0].score > 0
        assert results[0].metadata["doc_id"] == "doc1"
