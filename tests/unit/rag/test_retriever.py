import pytest
from scholar_lens.rag.retriever import HybridRetriever, RetrievalResult


class TestHybridRetriever:
    def test_instantiation(self):
        r = HybridRetriever()
        assert r is not None

    def test_rrf_fusion(self):
        r = HybridRetriever(rrf_k=60)
        bm25_results = [
            RetrievalResult(chunk_id="a", text="a", score=0.9, source="bm25", rank=1, metadata={}),
            RetrievalResult(chunk_id="b", text="b", score=0.7, source="bm25", rank=2, metadata={}),
        ]
        vector_results = [
            RetrievalResult(chunk_id="b", text="b", score=0.95, source="vector", rank=1, metadata={}),
            RetrievalResult(chunk_id="c", text="c", score=0.8, source="vector", rank=2, metadata={}),
        ]
        fused = r.rrf_fuse([bm25_results, vector_results])
        assert fused[0].chunk_id == "b"

    def test_rrf_single_source(self):
        r = HybridRetriever(rrf_k=60)
        results = [RetrievalResult(chunk_id="a", text="a", score=0.9, source="bm25", rank=1, metadata={})]
        fused = r.rrf_fuse([results])
        assert len(fused) == 1
        assert fused[0].chunk_id == "a"

    def test_empty_results(self):
        r = HybridRetriever(rrf_k=60)
        fused = r.rrf_fuse([[], []])
        assert fused == []


class TestRetrievalResult:
    def test_create(self):
        rr = RetrievalResult(chunk_id="x", text="hello", score=0.9, source="bm25", rank=1, metadata={"section_id": "1"})
        assert rr.chunk_id == "x"
        assert rr.source == "bm25"
