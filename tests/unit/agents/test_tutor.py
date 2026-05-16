import pytest
from unittest.mock import AsyncMock, MagicMock

from scholar_lens.agents.tutor import LearningTutorAgent
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import DocumentUnderstanding, Section


class TestLearningTutorAgent:
    @pytest.mark.asyncio
    async def test_respond_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "这是一个很好的问题。自注意力机制的核心思想是让序列中的每个位置都能关注到其他所有位置。"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = LearningTutorAgent(llm=mock_llm)
        state = ScholarLensState(
            doc_id="paper_001",
            section_id="3.1",
        )
        state.student_profile.level = "intermediate"
        state.add_message("user", "什么是自注意力机制？")

        result = await agent.respond(state)
        assert len(result.messages) > 1  # At least the user message + response
        assert result.current_step == "tutor"

    @pytest.mark.asyncio
    async def test_respond_with_mermaid_map(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Let me explain the structure."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = LearningTutorAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001", section_id="1")
        state.doc_understanding = DocumentUnderstanding(
            doc_type="research_paper",
            language="en",
            difficulty="advanced",
            estimated_reading_time=45,
            sections=[Section(section_id="1", title="Introduction", level=1)],
            mermaid_map="graph TD\n  A[Intro]-->B[Method]",
            key_terms=[],
        )
        state.add_message("user", "这篇论文的结构是什么？")

        result = await agent.respond(state)
        assert result.current_step == "tutor"

    @pytest.mark.asyncio
    async def test_respond_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

        agent = LearningTutorAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001")
        state.add_message("user", "Hello")

        result = await agent.respond(state)
        assert result.error != ""
