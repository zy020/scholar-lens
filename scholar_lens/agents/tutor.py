from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scholar_lens.agents.prompts import TUTOR_SYSTEM, TUTOR_RESPONSE
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.circuit_breaker import CircuitBreaker
from scholar_lens.core.exceptions import CircuitOpenError

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class LearningTutorAgent:
    """Learning Tutor Agent per spec Section 4.4.

    This is the only agent that directly converses with the student.
    Other agents serve through the tutor.

    Interaction modes: collaborative reading, Socratic questioning,
    scaffolding, teach-back, gap detection.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        core_memory_context: str = "",
    ) -> None:
        self._llm = llm
        self._core_memory_context = core_memory_context
        self._circuit_breaker = CircuitBreaker(name="llm-tutor")

    async def respond(self, state: ScholarLensState) -> ScholarLensState:
        """Generate a tutor response to the student's latest message."""
        state.current_step = "tutor"

        if not self._llm:
            state.error = "No LLM configured for Tutor"
            return state

        user_message = ""
        for msg in reversed(state.messages):
            if msg.get("role") == "user":
                user_message = msg["content"]
                break

        if not user_message:
            state.error = "No user message found"
            return state

        try:
            # Batch 5.2: detect gaps and provide collaborative hints
            gap_hint = self._detect_gaps(state, user_message)
            collab_hint = self._collaborative_hint(state)

            response_content = await self._call_llm(
                state, user_message, gap_hint=gap_hint, collab_hint=collab_hint,
            )
            state.add_message("assistant", response_content)

            # Batch 2: update knowledge tracing
            self._update_knowledge_tracing(state, user_message, response_content)
        except Exception as e:
            logger.error(f"Tutor response failed: {e}")
            state.error = f"Tutor response failed: {e}"

        return state

    # ===== Batch 5.2: Collaborative Reading + Gap Detection =====

    def _collaborative_hint(self, state: ScholarLensState) -> str:
        """When student enters a new section, provide a preview of key concepts."""
        if not state.doc_understanding or not state.section_id:
            return ""

        sections = state.doc_understanding.sections
        for sec in sections:
            if sec.section_id == state.section_id:
                l0 = state.doc_understanding.l0_summaries.get(sec.section_id, "")
                return (
                    f"The student is reading section '{sec.title}'. "
                    f"Brief summary: {l0}. "
                    f"Consider giving a preview of key concepts in this section."
                )
        return ""

    def _detect_gaps(self, state: ScholarLensState, user_message: str) -> str:
        """Detect knowledge gaps from the student's question."""
        if not state.doc_understanding:
            return ""

        prereqs = state.doc_understanding.prerequisites
        terms = {t.english.lower(): t for t in state.doc_understanding.key_terms}

        mentioned_terms = []
        for term_en, term_obj in terms.items():
            if term_en in user_message.lower():
                mentioned_terms.append(term_obj)

        weak_terms = [t for t in mentioned_terms if t.p_known < 0.4]
        if weak_terms:
            return (
                f"The student may have knowledge gaps in: "
                f"{', '.join(t.english for t in weak_terms)}. "
                f"Consider explaining these concepts with more scaffolding."
            )
        return ""

    # ===== Batch 2: Knowledge Tracing =====

    def _update_knowledge_tracing(
        self, state: ScholarLensState, question: str, answer: str,
    ) -> None:
        """Update p(known) for concepts mentioned in the interaction."""
        if not state.doc_understanding:
            return

        for term in state.doc_understanding.key_terms:
            if term.english.lower() in question.lower():
                # Student asked about this — slight boost for engagement
                term.p_known = min(1.0, term.p_known + 0.1)
            if term.english.lower() in answer.lower():
                # Tutor explained this — moderate boost
                term.p_known = min(1.0, term.p_known + 0.05)

        # Batch 2.2: propagate to related terms via SciERC relations
        self._propagate_knowledge(state.doc_understanding.key_terms)

    def _propagate_knowledge(self, terms: list) -> None:
        """Propagate p(known) through relation types.

        Hyponym-of: understanding "attention" boosts "multi-head attention"
        Used-for: understanding "softmax" boosts "attention calculation"
        Part-of: understanding "encoder" boosts "transformer"
        """
        for term in terms:
            if term.p_known >= 0.7 and term.relation_type:
                for related in terms:
                    if related is term:
                        continue
                    if not related.relation_type:
                        continue

                    # Propagate based on relation type
                    if (
                        term.relation_type == "Hyponym-of"
                        and term.english in related.english
                    ):
                        related.p_known = min(1.0, related.p_known + 0.05)
                    elif related.relation_type in ("Used-for", "Part-of"):
                        related.p_known = min(1.0, related.p_known + 0.03)

    async def _call_llm(
        self, state: ScholarLensState, user_message: str,
        gap_hint: str = "", collab_hint: str = "",
    ) -> str:
        if not await self._circuit_breaker.allow_request():
            raise CircuitOpenError("llm-tutor", self._circuit_breaker)
        try:
            mermaid_map = ""
            terms_context = ""
            if state.doc_understanding:
                mermaid_map = state.doc_understanding.mermaid_map
                # Batch 5.4: add term knowledge states to context
                if state.doc_understanding.key_terms:
                    term_lines = [
                        f"  {t.english}（{t.chinese}）p(known)={t.p_known:.0%}"
                        for t in state.doc_understanding.key_terms[:10]
                    ]
                    terms_context = "Student knowledge state:\n" + "\n".join(term_lines)

            core_memory = self._core_memory_context or ""
            if terms_context:
                core_memory += "\n\n" + terms_context

            retrieved_context = ""
            if state.retrieved_chunks:
                chunks_text = "\n".join(
                    f"[{c.get('section_id', '')}] {c.get('text', '')[:500]}"
                    for c in state.retrieved_chunks[:3]
                )
                retrieved_context = f"Retrieved context:\n{chunks_text}"

            if state.explanation_result:
                retrieved_context += f"\n\nExplanation from Explainer:\n{state.explanation_result.explanation}"

            # Inject gap hints and collaborative hints
            hints = ""
            if gap_hint:
                hints += f"\nGap detection: {gap_hint}"
            if collab_hint:
                hints += f"\nCollaborative reading: {collab_hint}"

            system_prompt = TUTOR_SYSTEM.format(
                core_memory=core_memory or "No core memory loaded",
                mermaid_map=mermaid_map or "No document structure available",
            )

            user_prompt = TUTOR_RESPONSE.format(
                question=user_message,
                section_id=state.section_id or "unknown",
                student_level=state.student_profile.level,
                retrieved_context=(retrieved_context + hints) or "No retrieved context available",
            )

            history = state.messages[-10:] if len(state.messages) > 10 else state.messages
            langchain_messages = [SystemMessage(content=system_prompt)]
            for msg in history:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            langchain_messages.append(HumanMessage(content=user_prompt))

            response = await asyncio.wait_for(
                self._llm.ainvoke(langchain_messages), timeout=120,
            )
            await self._circuit_breaker.record_success()
            return response.content
        except CircuitOpenError:
            raise
        except Exception:
            await self._circuit_breaker.record_failure()
            raise
