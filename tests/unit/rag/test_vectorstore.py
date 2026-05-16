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
            Chunk(chunk_id="doc1_1_0", text="Self-attention mechanism allows each position to attend to all other positions.", metadata=ChunkMetadata(section_id="3.1", section_type="method", doc_id="doc1"), layer="L2"),
            Chunk(chunk_id="doc1_2_0", text="The training procedure uses Adam optimizer with learning rate warmup.", metadata=ChunkMetadata(section_id="4", section_type="method", doc_id="doc1"), layer="L2"),
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
            Chunk(chunk_id="doc1_1_0", text="Hello", metadata=ChunkMetadata(section_id="1", section_type="prose", doc_id="doc1"), layer="L2"),
        ]
        vs.add_chunks(chunks, embeddings=[[0.1] * 10])
        vs.delete_by_doc_id("doc1")
        results = vs.query(query_embedding=[0.1] * 10, top_k=5)
        assert len(results) == 0
