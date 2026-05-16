import pytest
from unittest.mock import AsyncMock, MagicMock
from scholar_lens.agents.validator import ValidatorAgent
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import ExplanationResult, ValidationResult


class TestValidatorAgent:
    @pytest.mark.asyncio
    async def test_validate_passed(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''```json
{"passed": true, "confidence": "high", "issues": [], "correction": null}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = ValidatorAgent(llm=mock_llm)
        state = ScholarLensState()
        state.explanation_result = ExplanationResult(original="Self-attention computes attention", translation="自注意力计算注意力", explanation="自注意力是一种机制", confidence="high")

        result = await agent.validate(state)
        assert result.validation_result is not None
        assert result.validation_result.passed is True

    @pytest.mark.asyncio
    async def test_validate_failed_with_correction(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''```json
{"passed": false, "confidence": "low", "issues": ["Term 'attention' mistranslated as 关注 instead of 注意力"], "correction": "attention should be 注意力, not 关注"}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = ValidatorAgent(llm=mock_llm)
        state = ScholarLensState()
        state.explanation_result = ExplanationResult(original="test", translation="test", explanation="test", confidence="low")

        result = await agent.validate(state)
        assert result.validation_result is not None
        assert result.validation_result.passed is False
        assert result.validation_result.correction is not None

    @pytest.mark.asyncio
    async def test_validate_skips_when_no_explanation(self):
        agent = ValidatorAgent(llm=None)
        state = ScholarLensState()
        result = await agent.validate(state)
        assert result.validation_result is None
        assert result.current_step == "validate"

    @pytest.mark.asyncio
    async def test_validate_failure_does_not_block(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        agent = ValidatorAgent(llm=mock_llm)
        state = ScholarLensState()
        state.explanation_result = ExplanationResult(original="x", translation="x", explanation="x", confidence="medium")

        result = await agent.validate(state)
        assert result.validation_result is not None
        assert result.validation_result.confidence == "unverified"
