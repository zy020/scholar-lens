from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from scholar_lens.memory.core_memory import CoreMemory
from scholar_lens.memory.document_memory import DocumentMemory
from scholar_lens.memory.reflection_memory import ReflectionMemory
from scholar_lens.memory.structured_memory import StructuredMemory


class MemoryManager:
    """Unified memory manager coordinating all four tiers."""

    def __init__(self, data_dir: str = "data") -> None:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        self.core_memory = CoreMemory()
        self.structured = StructuredMemory(db_path=str(data_path / "memory.db"))
        self.reflection = ReflectionMemory(knowledge_dir=str(data_path / "knowledge"))
        self.document = DocumentMemory()
        self._recent_actions: list[str] = []

    async def close(self) -> None:
        await self.structured.close()

    def get_core_context(self) -> str:
        return self.core_memory.to_context_string()

    async def get_personalization_context(self, doc_id: str = "") -> str:
        parts = [self.get_core_context()]
        concepts = await self.structured.get_concept_memory(doc_id=doc_id, limit=8)
        if concepts:
            formatted = [
                f"{item['concept']} ({item['status']}, evidence={item['evidence_count']})"
                for item in concepts[:8]
            ]
            parts.append("Learning Concepts: " + "; ".join(formatted))
        return "\n".join(part for part in parts if part.strip())

    async def get_snapshot(self, doc_id: str = "", limit: int = 30) -> dict[str, Any]:
        return {
            "core": {
                "student_profile": self.core_memory.student_profile,
                "current_position": self.core_memory.current_position,
                "active_glossary": list(self.core_memory.active_glossary),
                "session_summary": self.core_memory.session_summary,
            },
            "recent_events": await self.structured.get_learning_events(doc_id=doc_id, limit=limit),
            "concepts": await self.structured.get_concept_memory(doc_id=doc_id, limit=limit),
            "document": {
                "doc_id": self.document.doc_id,
                "loaded": bool(self.document.doc_id),
            },
        }

    async def get_retrieval_hints(self, doc_id: str = "") -> dict[str, Any]:
        current_section_id = ""
        current = self.core_memory.current_position
        if current and ":" in current:
            current_doc_id, section_id = current.split(":", 1)
            if not doc_id or current_doc_id == doc_id:
                current_section_id = section_id
        concepts = await self.structured.get_concept_memory(doc_id=doc_id, limit=12)
        weak_concepts = [
            str(item["concept"])
            for item in concepts
            if item.get("status") in {"needs_review", "learning"}
        ]
        return {
            "current_section_id": current_section_id,
            "concepts": weak_concepts[:8],
        }

    async def clear_session_memory(self) -> None:
        self._recent_actions.clear()
        self.core_memory.session_summary = ""
        await self.structured.clear_session_memory()

    async def clear_document_memory(self, doc_id: str = "") -> None:
        if not doc_id or self.document.doc_id == doc_id:
            self.document.clear()
        if doc_id and self.core_memory.current_position.startswith(f"{doc_id}:"):
            self.core_memory.current_position = ""
        await self.structured.clear_document_memory(doc_id)

    async def clear_all_memory(self) -> None:
        self.core_memory = CoreMemory()
        self.document.clear()
        self._recent_actions.clear()
        await self.structured.clear_all_memory()

    async def record_event(
        self,
        event_type: str,
        doc_id: str = "",
        section_id: str = "",
        payload: dict[str, Any] | None = None,
        summary_llm=None,
    ) -> None:
        from scholar_lens.memory.memory_graph import run_memory_update_graph

        await run_memory_update_graph(
            self,
            event_type=event_type,
            doc_id=doc_id,
            section_id=section_id,
            payload=payload or {},
            summary_llm=summary_llm,
        )

    def _update_core_from_event(
        self,
        event_type: str,
        doc_id: str,
        section_id: str,
        payload: dict[str, Any],
    ) -> None:
        if doc_id and section_id:
            self.core_memory.update_position(doc_id, section_id)
        action = self._action_summary(event_type, payload)
        if action:
            self._recent_actions.append(action)
            self._recent_actions = self._recent_actions[-8:]
            self.core_memory.session_summary = "Recent learning actions: " + " | ".join(self._recent_actions)

    def _action_summary(self, event_type: str, payload: dict[str, Any]) -> str:
        if event_type == "section_read":
            title = str(payload.get("title") or "section").strip()
            return f"Read {title[:80]}"
        if event_type == "chat_question":
            message = str(payload.get("message") or "").strip()
            return f"Asked: {message[:120]}" if message else "Asked a question"
        if event_type == "translate_text":
            return "Translated selected text"
        if event_type == "explain_text":
            return "Explained selected text"
        if event_type in {"brief_generate", "brief_view"}:
            return "Reviewed study brief"
        if event_type == "export_obsidian":
            return "Exported learning notes"
        return ""

    def _concept_status(self, event_type: str, payload: dict[str, Any]) -> str:
        text = " ".join(str(value) for value in payload.values()).lower()
        if event_type in {"explain_text", "translate_text"}:
            return "learning"
        if re.search(r"不理解|不会|困惑|看不懂|confus|struggl|formula|公式|推导", text):
            return "needs_review"
        if event_type == "section_read":
            return "seen"
        return "learning"

    def _extract_concepts(self, event_type: str, payload: dict[str, Any]) -> list[str]:
        if event_type not in {"chat_question", "explain_text", "translate_text", "section_read"}:
            return []
        text = " ".join(
            str(payload.get(key) or "")
            for key in ("message", "text_preview", "title")
        )
        if not text.strip():
            return []
        normalized = text.lower()
        known_phrases = [
            "self-attention",
            "positional encoding",
            "graph neural network",
            "attention formula",
            "query key value",
            "multi-head attention",
            "transformer architecture",
        ]
        concepts = [phrase for phrase in known_phrases if phrase in normalized]
        stopwords = {
            "what", "is", "are", "the", "a", "an", "of", "and", "or", "to", "in", "on",
            "how", "why", "does", "do", "this", "that", "with", "for", "about", "explain",
            "please", "use", "used", "here", "there",
        }
        raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", normalized)
        groups: list[list[str]] = []
        current: list[str] = []
        for token in raw_tokens:
            if token in stopwords or len(token) <= 2:
                if current:
                    groups.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            groups.append(current)
        for group in groups:
            for width in range(min(3, len(group)), 0, -1):
                for idx in range(0, len(group) - width + 1):
                    phrase = " ".join(group[idx: idx + width])
                    if phrase not in concepts and (width > 1 or "-" in phrase or len(phrase) >= 4):
                        concepts.append(phrase)
                if concepts:
                    break
        return concepts[:8]
