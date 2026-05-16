from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scholar_lens.agents.prompts import TUTOR_SYSTEM, TUTOR_RESPONSE
from scholar_lens.agents.state import ScholarLensState

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

    async def respond(self, state: ScholarLensState) -> ScholarLensState:
        """Generate a tutor response to the student's latest message."""
        state.current_step = "tutor"

        if not self._llm:
            state.error = "No LLM configured for Tutor"
            return state

        # Get the latest user message
        user_message = ""
        for msg in reversed(state.messages):
            if msg.get("role") == "user":
                user_message = msg["content"]
                break

        if not user_message:
            state.error = "No user message found"
            return state

        try:
            response_content = await self._call_llm(state, user_message)
            state.add_message("assistant", response_content)
        except Exception as e:
            logger.error(f"Tutor response failed: {e}")
            state.error = f"Tutor response failed: {e}"

        return state

    async def _call_llm(self, state: ScholarLensState, user_message: str) -> str:
        # Build context
        mermaid_map = ""
        if state.doc_understanding:
            mermaid_map = state.doc_understanding.mermaid_map

        core_memory = self._core_memory_context or ""

        # Build retrieved context
        retrieved_context = ""
        if state.retrieved_chunks:
            chunks_text = "\n".join(
                f"[{c.get('section_id', '')}] {c.get('text', '')[:500]}"
                for c in state.retrieved_chunks[:3]
            )
            retrieved_context = f"Retrieved context:\n{chunks_text}"

        # Include explanation result if available
        if state.explanation_result:
            retrieved_context += f"\n\nExplanation from Explainer:\n{state.explanation_result.explanation}"

        system_prompt = TUTOR_SYSTEM.format(
            core_memory=core_memory or "No core memory loaded",
            mermaid_map=mermaid_map or "No document structure available",
        )

        user_prompt = TUTOR_RESPONSE.format(
            question=user_message,
            section_id=state.section_id or "unknown",
            student_level=state.student_profile.level,
            retrieved_context=retrieved_context or "No retrieved context available",
        )

        # Build conversation history (last 10 messages)
        history = state.messages[-10:] if len(state.messages) > 10 else state.messages
        langchain_messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        langchain_messages.append(HumanMessage(content=user_prompt))

        response = await self._llm.ainvoke(langchain_messages)
        return response.content
