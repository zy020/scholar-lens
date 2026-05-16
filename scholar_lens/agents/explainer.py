from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from scholar_lens.agents.prompts import EXPLAINER_SYSTEM, EXPLAINER_TRANSLATE
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import ExplanationResult, Term

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ContentExplainerAgent:
    """Content Explainer Agent per spec Section 4.2."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    async def explain(self, state: ScholarLensState) -> ScholarLensState:
        if not self._llm:
            state.error = "No LLM configured for Explainer"
            state.current_step = "explain"
            return state

        if not state.explanation_request:
            state.error = "No explanation request in state"
            state.current_step = "explain"
            return state

        try:
            result_text = await self._call_llm(state)
            explanation = self._parse_result(result_text)
            state.explanation_result = explanation
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            state.error = f"Explanation failed: {e}"

        state.current_step = "explain"
        return state

    async def _call_llm(self, state: ScholarLensState) -> str:
        section_title = ""
        if state.doc_understanding:
            for s in state.doc_understanding.sections:
                if s.section_id == state.section_id:
                    section_title = s.title
                    break

        previous_count = sum(1 for m in state.messages if m.get("role") == "assistant")

        prompt = EXPLAINER_TRANSLATE.format(
            level=state.student_profile.level,
            section_title=section_title or state.section_id,
            previous_count=previous_count,
            target_text=state.explanation_request,
        )

        response = await self._llm.ainvoke([
            SystemMessage(content=EXPLAINER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content

    def _parse_result(self, llm_output: str) -> ExplanationResult:
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", llm_output, re.DOTALL)
        json_str = json_match.group(1) if json_match else llm_output

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return ExplanationResult(original=llm_output, translation="", explanation=llm_output, confidence="unverified")

        related_terms = [Term(english=t.get("english", ""), chinese=t.get("chinese", "")) for t in data.get("related_terms", [])]

        return ExplanationResult(
            original=data.get("original", ""), translation=data.get("translation", ""),
            explanation=data.get("explanation", ""), related_terms=related_terms,
            difficulty_level=data.get("difficulty_level", "intermediate"),
            source_section=data.get("source_section", ""), confidence=data.get("confidence", "medium"),
        )
