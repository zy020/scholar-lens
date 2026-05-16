from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from scholar_lens.agents.prompts import VALIDATOR_SYSTEM, VALIDATOR_CHECK
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import ValidationResult

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """Validator Agent per spec Section 4.3. Never blocks main flow."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    async def validate(self, state: ScholarLensState) -> ScholarLensState:
        state.current_step = "validate"
        if state.explanation_result is None:
            return state

        rule_result = self._rule_validate(state)
        if not rule_result.passed:
            state.validation_result = rule_result
            return state

        if self._llm:
            try:
                llm_result = await self._llm_validate(state)
                state.validation_result = llm_result
            except Exception as e:
                logger.warning(f"LLM validation failed: {e}")
                state.validation_result = ValidationResult(passed=True, confidence="unverified", issues=[f"LLM validation skipped: {e}"])
        else:
            state.validation_result = ValidationResult(passed=True, confidence="unverified", issues=["No LLM available for validation"])

        return state

    def _rule_validate(self, state: ScholarLensState) -> ValidationResult:
        issues = []
        explanation = state.explanation_result
        if not explanation:
            return ValidationResult(passed=True, confidence="high")
        if not explanation.translation.strip():
            issues.append("Empty translation")
        if not explanation.explanation.strip():
            issues.append("Empty explanation")
        if issues:
            return ValidationResult(passed=False, confidence="high", issues=issues)
        return ValidationResult(passed=True, confidence="high")

    async def _llm_validate(self, state: ScholarLensState) -> ValidationResult:
        explanation = state.explanation_result
        source_text = ""
        for msg in state.messages:
            if "Document text:" in msg.get("content", ""):
                source_text = msg["content"][:2000]
                break

        prompt = VALIDATOR_CHECK.format(
            source_text=source_text or "Source not available",
            explanation=f"Original: {explanation.original}\nTranslation: {explanation.translation}\nExplanation: {explanation.explanation}",
        )

        response = await self._llm.ainvoke([SystemMessage(content=VALIDATOR_SYSTEM), HumanMessage(content=prompt)])
        return self._parse_result(response.content)

    def _parse_result(self, llm_output: str) -> ValidationResult:
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", llm_output, re.DOTALL)
        json_str = json_match.group(1) if json_match else llm_output

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return ValidationResult(passed=True, confidence="unverified")

        return ValidationResult(
            passed=data.get("passed", True), confidence=data.get("confidence", "medium"),
            issues=data.get("issues", []), correction=data.get("correction"),
        )
