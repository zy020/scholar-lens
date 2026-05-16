import pytest
from unittest.mock import AsyncMock, MagicMock
from scholar_lens.agents.doc_analyzer import DocumentAnalyzerAgent
from scholar_lens.agents.state import ScholarLensState


class TestDocumentAnalyzerAgent:
    @pytest.mark.asyncio
    async def test_analyze_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''```json
{
    "doc_type": "research_paper",
    "language": "en",
    "difficulty": "advanced",
    "estimated_reading_time": 45,
    "sections": [{"section_id": "1", "title": "Introduction", "level": 1, "page_start": 1, "page_end": 2, "section_type": "prose", "difficulty": "intermediate"}],
    "mermaid_map": "graph TD\\n  A[Intro]-->B[Method]",
    "key_terms": [{"english": "transformer", "chinese": "Transformer"}],
    "l0_summaries": {"1": "Introduces the problem"},
    "l1_overviews": {"1": "This paper introduces the Transformer architecture..."},
    "references": [],
    "citation_contexts": [],
    "prerequisites": ["attention mechanism"]
}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = DocumentAnalyzerAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001", file_path="test.pdf")
        state.add_message("system", "Document text: Attention Is All You Need. Abstract: We propose a new architecture...")

        result = await agent.analyze(state)
        assert result.doc_understanding is not None
        assert result.doc_understanding.doc_type == "research_paper"
        assert result.current_step == "analyze"

    @pytest.mark.asyncio
    async def test_analyze_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        agent = DocumentAnalyzerAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001")
        state.add_message("system", "Document text: Some text")

        result = await agent.analyze(state)
        assert result.error != ""
