import pytest
from unittest.mock import AsyncMock, MagicMock
from scholar_lens.agents.explainer import ContentExplainerAgent
from scholar_lens.agents.state import ScholarLensState


class TestContentExplainerAgent:
    @pytest.mark.asyncio
    async def test_explain_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''```json
{
    "original": "The self-attention mechanism computes attention scores",
    "translation": "自注意力机制（self-attention mechanism）计算注意力分数",
    "explanation": "自注意力是一种让序列中每个位置都能关注其他位置的机制。",
    "related_terms": [{"english": "attention", "chinese": "注意力"}],
    "difficulty_level": "intermediate",
    "source_section": "3.1",
    "confidence": "high"
}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = ContentExplainerAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001", section_id="3.1", explanation_request="Explain self-attention")
        state.student_profile.level = "intermediate"

        result = await agent.explain(state)
        assert result.explanation_result is not None
        assert result.explanation_result.confidence == "high"
        assert result.current_step == "explain"

    @pytest.mark.asyncio
    async def test_explain_fallback(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

        agent = ContentExplainerAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001", section_id="3.1", explanation_request="Explain this")

        result = await agent.explain(state)
        assert result.error != ""
