from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from scholar_lens.rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)

_SECTION_WEIGHTS = {
    "method": 1.2,
    "results": 1.1,
    "prose": 1.0,
    "citation_context": 0.9,
    "references": 0.3,
}

_DIFFICULTY_TARGETS = {
    "beginner": 0.3,
    "intermediate": 0.5,
    "advanced": 0.8,
}


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, results: list[RetrievalResult], query: str = "", student_level: str = "intermediate") -> list[RetrievalResult]:
        ...


class RuleReranker(BaseReranker):
    """Level 3: Rule-based reranking. Zero cost, zero latency."""

    def rerank(self, results: list[RetrievalResult], query: str = "", student_level: str = "intermediate") -> list[RetrievalResult]:
        if not results:
            return []
        target_difficulty = _DIFFICULTY_TARGETS.get(student_level, 0.5)
        scored = []
        for r in results:
            score = r.score
            section_type = r.metadata.get("section_type", "prose")
            score *= _SECTION_WEIGHTS.get(section_type, 1.0)
            difficulty = r.metadata.get("difficulty_score", 0.5)
            diff_penalty = abs(difficulty - target_difficulty) * 0.5
            score *= (1.0 - diff_penalty)
            cross_refs = r.metadata.get("cross_refs", [])
            if isinstance(cross_refs, list) and cross_refs:
                score *= 1.05
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(chunk_id=r.chunk_id, text=r.text, score=score, source="rule_reranked", rank=rank, metadata=r.metadata)
            for rank, (score, r) in enumerate(scored, 1)
        ]


class RerankerPipeline:
    """4-level reranking pipeline per spec Section 6.6."""

    def __init__(self, model_reranker: BaseReranker | None = None, llm_reranker: BaseReranker | None = None) -> None:
        self._model_reranker = model_reranker
        self._llm_reranker = llm_reranker
        self._rule_reranker = RuleReranker()
        self.level = self._determine_level()

    def _determine_level(self) -> str:
        if self._model_reranker:
            return "model"
        if self._llm_reranker:
            return "llm"
        return "rule"

    def rerank(self, results: list[RetrievalResult], query: str = "", student_level: str = "intermediate") -> list[RetrievalResult]:
        if self._model_reranker:
            try:
                return self._model_reranker.rerank(results, query, student_level)
            except Exception as e:
                logger.warning(f"Model reranker failed: {e}, falling back")
        if self._llm_reranker:
            try:
                return self._llm_reranker.rerank(results, query, student_level)
            except Exception as e:
                logger.warning(f"LLM reranker failed: {e}, falling back")
        return self._rule_reranker.rerank(results, query, student_level)
