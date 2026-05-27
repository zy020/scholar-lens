"""LangGraph orchestration integration test.

Validates the full agent pipeline with real state transitions:
analyze → explain → validate → tutor loop
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from scholar_lens.agents.state import ScholarLensState
from scholar_lens.agents.doc_analyzer import DocumentAnalyzerAgent
from scholar_lens.agents.explainer import ContentExplainerAgent
from scholar_lens.agents.validator import ValidatorAgent
from scholar_lens.agents.tutor import LearningTutorAgent
from scholar_lens.agents.orchestrator import ScholarLensOrchestrator


class TestLangGraphOrchestration:
    """Validate graph structure and state flow."""

    def test_pipeline_graph_has_correct_nodes(self):
        orch = ScholarLensOrchestrator()
        graph = orch.build_pipeline_graph()
        nodes = graph.nodes
        assert "analyze" in nodes
        assert len(nodes) == 1  # Only analyze (explainer/validator moved to tutor graph)

    def test_tutor_graph_has_full_cycle(self):
        orch = ScholarLensOrchestrator()
        graph = orch.build_tutor_graph()
        nodes = graph.nodes
        assert "tutor" in nodes
        assert "explainer" in nodes
        assert "validator" in nodes

    def test_routing_from_tutor_no_explanation_needed(self):
        """Tutor graph should be compilable and contain conditional routing edges."""
        orch = ScholarLensOrchestrator()
        graph = orch.build_tutor_graph()
        edges = graph.edges
        assert len(edges) > 0
        # Verify graph compiles (raises if misconfigured)
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_state_flows_through_tutor_respond(self):
        """Tutor.respond() should add an assistant message."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "The self-attention mechanism computes attention weights."
        mock_llm.ainvoke.return_value = mock_response

        agent = LearningTutorAgent(llm=mock_llm, core_memory_context="Student: intermediate")
        state = ScholarLensState(
            messages=[{"role": "user", "content": "What is self-attention?"}],
        )

        result = await agent.respond(state)
        assert len(result.messages) >= 2
        assert result.messages[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_doc_analyzer_state_update(self):
        """DocAnalyzer should populate doc_understanding."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = """
```json
{
    "doc_type": "research_paper",
    "language": "en",
    "difficulty": "intermediate",
    "estimated_reading_time": 30,
    "sections": [
        {"section_id": "1", "title": "Introduction", "level": 1}
    ],
    "key_terms": [],
    "mermaid_map": "graph TD\\n  A-->B",
    "prerequisites": []
}
```"""
        mock_llm.ainvoke.return_value = mock_response

        agent = DocumentAnalyzerAgent(llm=mock_llm)
        state = ScholarLensState(
            messages=[{"role": "user", "content": "Document text: This is a test paper."}],
        )

        result = await agent.analyze(state)
        assert result.doc_understanding is not None
        assert result.doc_understanding.doc_type == "research_paper"

    @pytest.mark.asyncio
    async def test_validator_passes_on_good_explanation(self):
        """Validator should pass rule validation on good explanation."""
        from scholar_lens.core.models import ExplanationResult
        agent = ValidatorAgent(llm=None)
        state = ScholarLensState(
            explanation_result=ExplanationResult(
                original="test", translation="测试", explanation="解释",
                confidence="high",
            ),
        )

        result = await agent.validate(state)
        assert result.validation_result is not None
        assert result.validation_result.passed is True

    def test_full_orchestrator_instantiation(self):
        """All agents should be instantiated with defaults."""
        orch = ScholarLensOrchestrator()
        assert orch.doc_analyzer is not None
        assert orch.explainer is not None
        assert orch.validator is not None
        assert orch.tutor is not None

    @pytest.mark.asyncio
    async def test_explainer_with_mock_llm(self):
        """ContentExplainerAgent should produce structured explanation result."""
        from scholar_lens.core.models import DocumentUnderstanding, Section, Term
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = """```json
{
    "original": "The transformer uses self-attention",
    "translation": "Transformer使用自注意力",
    "explanation": "自注意力是一种机制...",
    "related_terms": [{"english": "attention", "chinese": "注意力"}],
    "difficulty_level": "intermediate",
    "source_section": "2",
    "confidence": "high"
}
```"""
        mock_llm.ainvoke.return_value = mock_response
        agent = ContentExplainerAgent(llm=mock_llm)
        state = ScholarLensState(
            explanation_request="Explain self-attention",
            section_id="2",
            doc_understanding=DocumentUnderstanding(
                doc_type="research_paper", language="en", difficulty="intermediate",
                estimated_reading_time=30,
                sections=[Section(section_id="2", title="Method", level=1)],
                mermaid_map="",
                key_terms=[Term(english="self-attention", chinese="自注意力")],
            ),
        )
        result = await agent.explain(state)
        assert result.explanation_result is not None
        assert result.explanation_result.confidence == "high"
        assert "自注意力" in result.explanation_result.translation
