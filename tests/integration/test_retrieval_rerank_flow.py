"""Integration test: Retrieval → Reranking flow."""

import pytest
from scholar_lens.rag.retriever import HybridRetriever, RetrievalResult
from scholar_lens.rag.reranker import RuleReranker, RerankerPipeline
from scholar_lens.parsers.models import Chunk, ChunkMetadata


def _make_result(chunk_id: str, text: str, score: float, source: str, rank: int, section_type: str = "prose", difficulty: float = 0.5, cross_refs: list | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source=source,
        rank=rank,
        metadata={
            "section_id": chunk_id.split("_")[0],
            "section_type": section_type,
            "difficulty_score": difficulty,
            "cross_refs": cross_refs or [],
        },
    )


class TestRetrievalRerankFlow:
    def test_bm25_vector_fusion_flow(self):
        """BM25 + vector results should fuse correctly via RRF."""
        bm25_results = [
            _make_result("1_0", "transformer architecture", 0.9, "bm25", 1),
            _make_result("2_0", "attention mechanism", 0.7, "bm25", 2),
        ]
        vector_results = [
            _make_result("2_0", "attention mechanism", 0.95, "vector", 1),
            _make_result("1_0", "transformer architecture", 0.8, "vector", 2),
        ]
        retriever = HybridRetriever(rrf_k=60)
        fused = retriever.rrf_fuse([bm25_results, vector_results], top_k=2)
        assert len(fused) == 2
        assert fused[0].source == "rrf_fused"
        # chunk 2_0 should rank higher because it's #1 in vector and #2 in BM25
        # RRF: 1/(60+1) + 1/(60+2) ≈ 0.0325 for chunk 2_0
        # vs 1/(60+1) + 1/(60+2) = same for chunk 1_0
        # Actually they tie since both have rank 1 and 2 in different sources

    def test_fusion_then_rerank_flow(self):
        """Reranker should re-score fused results."""
        bm25_results = [
            _make_result("3_0", "method description", 0.8, "bm25", 1, section_type="method", difficulty=0.7),
            _make_result("1_0", "intro text", 0.6, "bm25", 2, section_type="prose", difficulty=0.3),
        ]
        vector_results = [
            _make_result("1_0", "intro text", 0.9, "vector", 1, section_type="prose", difficulty=0.3),
            _make_result("3_0", "method description", 0.7, "vector", 2, section_type="method", difficulty=0.7),
        ]
        retriever = HybridRetriever()
        fused = retriever.rrf_fuse([bm25_results, vector_results])

        reranker = RuleReranker()
        reranked = reranker.rerank(fused, student_level="advanced")
        assert len(reranked) == 2
        assert reranked[0].source == "rule_reranked"
        # Method section should rank higher for advanced student
        assert reranked[0].chunk_id == "3_0"

    def test_reranker_fallback_chain(self):
        """When model reranker fails, should fall back to rule."""
        results = [_make_result("1_0", "text", 0.8, "vector", 1)]
        pipeline = RerankerPipeline(model_reranker=None, llm_reranker=None)
        assert pipeline.level == "rule"
        reranked = pipeline.rerank(results, student_level="beginner")
        assert len(reranked) == 1
        assert reranked[0].source == "rule_reranked"

    def test_empty_results_handled(self):
        """Empty results should be handled gracefully."""
        retriever = HybridRetriever()
        fused = retriever.rrf_fuse([], [])
        assert fused == []

        reranker = RuleReranker()
        reranked = reranker.rerank([])
        assert reranked == []
