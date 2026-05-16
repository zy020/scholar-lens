import pytest
from scholar_lens.rag.reranker import RuleReranker, RerankerPipeline
from scholar_lens.rag.retriever import RetrievalResult


class TestRuleReranker:
    def test_section_type_relevance(self):
        reranker = RuleReranker()
        results = [
            RetrievalResult(chunk_id="a", text="a", score=0.5, source="rrf", rank=1, metadata={"section_type": "method"}),
            RetrievalResult(chunk_id="b", text="b", score=0.5, source="rrf", rank=2, metadata={"section_type": "references"}),
        ]
        reranked = reranker.rerank(results, student_level="intermediate")
        assert reranked[0].metadata.get("section_type") == "method"

    def test_difficulty_match(self):
        reranker = RuleReranker()
        results = [
            RetrievalResult(chunk_id="a", text="a", score=0.5, source="rrf", rank=1, metadata={"difficulty_score": 0.9, "section_type": "prose"}),
            RetrievalResult(chunk_id="b", text="b", score=0.5, source="rrf", rank=2, metadata={"difficulty_score": 0.5, "section_type": "prose"}),
        ]
        reranked = reranker.rerank(results, student_level="intermediate")
        assert reranked[0].chunk_id == "b"

    def test_empty_results(self):
        reranker = RuleReranker()
        reranked = reranker.rerank([], student_level="beginner")
        assert reranked == []


class TestRerankerPipeline:
    def test_no_reranker_falls_back_to_rule(self):
        pipeline = RerankerPipeline()
        assert pipeline.level == "rule"

    def test_rule_reranker_always_available(self):
        pipeline = RerankerPipeline()
        results = [
            RetrievalResult(chunk_id="a", text="a", score=0.5, source="rrf", rank=1, metadata={"section_type": "prose", "difficulty_score": 0.5}),
        ]
        reranked = pipeline.rerank(results, student_level="intermediate")
        assert len(reranked) == 1
