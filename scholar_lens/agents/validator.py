from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from scholar_lens.agents.prompts import VALIDATOR_SYSTEM, VALIDATOR_CHECK
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.circuit_breaker import CircuitBreaker
from scholar_lens.core.exceptions import CircuitOpenError
from scholar_lens.core.models import ValidationResult
from scholar_lens.core.utils import extract_json_from_llm_output

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """Validator Agent per spec Section 4.3. Never blocks main flow.

    Trigger conditions:
    - Rule-based (term consistency): always
    - LLM hallucination detection: first-time explanation, advanced sections,
      student-flagged, or 10% random sampling for quality monitoring
    """

    def __init__(self, llm: BaseChatModel | None = None, sampling_rate: float = 0.10) -> None:
        self._llm = llm
        self._sampling_rate = sampling_rate
        self._circuit_breaker = CircuitBreaker(name="llm-validator")

    async def validate(self, state: ScholarLensState) -> ScholarLensState:
        state.current_step = "validate"
        if state.explanation_result is None:
            return state

        # Rule validation always runs (zero cost)
        rule_result = self._rule_validate(state)
        if not rule_result.passed:
            state.validation_result = rule_result
            return state

        # LLM validation: trigger conditions + random sampling
        should_validate = (
            state.explanation_result.confidence in ("unverified", "low")
            or random.random() < self._sampling_rate  # Batch 5.3: 10% sampling
        )

        if self._llm and should_validate:
            try:
                llm_result = await self._llm_validate(state)
                state.validation_result = llm_result
            except Exception as e:
                logger.warning(f"LLM validation failed: {e}")
                state.validation_result = ValidationResult(
                    passed=False, confidence="unverified",
                    issues=[f"LLM validation error: {e}"],
                )
        elif not self._llm:
            state.validation_result = ValidationResult(
                passed=True,
                confidence="unverified",
                issues=["LLM validation skipped — rule check passed"],
            )
        else:
            state.validation_result = ValidationResult(passed=True, confidence="unverified", issues=["LLM validation skipped — rule check passed"])

        return state

    def _rule_validate(self, state: ScholarLensState) -> ValidationResult:
        issues = []
        explanation = state.explanation_result
        if not explanation.translation.strip():
            issues.append("Empty translation")
        if not explanation.explanation.strip():
            issues.append("Empty explanation")
        if issues:
            return ValidationResult(passed=False, confidence="high", issues=issues)
        return ValidationResult(passed=True, confidence="high")

    async def _llm_validate(self, state: ScholarLensState) -> ValidationResult:
        if not await self._circuit_breaker.allow_request():
            return ValidationResult(passed=False, confidence="unverified", issues=["LLM circuit breaker open — validation skipped"])
        try:
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

            response = await asyncio.wait_for(
                self._llm.ainvoke([SystemMessage(content=VALIDATOR_SYSTEM), HumanMessage(content=prompt)]), timeout=60,
            )
            await self._circuit_breaker.record_success()
            return self._parse_result(response.content)
        except Exception:
            await self._circuit_breaker.record_failure()
            return ValidationResult(passed=False, confidence="unverified", issues=["LLM validation failed — result unverified"])

    def _parse_result(self, llm_output: str) -> ValidationResult:
        data = extract_json_from_llm_output(llm_output)
        if not data:
            return ValidationResult(passed=False, confidence="unverified", issues=["Failed to parse LLM validation output"])

        return ValidationResult(
            passed=data.get("passed", True), confidence=data.get("confidence", "medium"),
            issues=data.get("issues", []), correction=data.get("correction"),
        )
