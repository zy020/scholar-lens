from __future__ import annotations

from pydantic import BaseModel, Field


class CoreMemory(BaseModel):
    """Tier 1: Core memory — always in agent context, ~500 token budget.

    - Student profile (3-5 sentences)
    - Current reading position (doc + section)
    - Active term glossary (last 10-20 terms)
    - Current session summary (1-2 paragraphs)
    """

    student_profile: str = ""
    current_position: str = ""  # "doc_id:section_id"
    active_glossary: list[str] = Field(default_factory=list)  # ["term:translation", ...]
    session_summary: str = ""

    _MAX_GLOSSARY_SIZE = 20

    def model_post_init(self, __context) -> None:
        self.active_glossary = self.active_glossary[-self._MAX_GLOSSARY_SIZE:]

    def update_position(self, doc_id: str, section_id: str) -> None:
        self.current_position = f"{doc_id}:{section_id}"

    def add_glossary_entry(self, term: str, translation: str) -> None:
        entry = f"{term}:{translation}"
        self.active_glossary = [e for e in self.active_glossary if not e.startswith(f"{term}:")]
        self.active_glossary.append(entry)
        if len(self.active_glossary) > self._MAX_GLOSSARY_SIZE:
            self.active_glossary = self.active_glossary[-self._MAX_GLOSSARY_SIZE:]

    def to_context_string(self) -> str:
        parts = []
        if self.student_profile:
            parts.append(f"Student Profile: {self.student_profile}")
        if self.current_position:
            parts.append(f"Current Position: {self.current_position}")
        if self.active_glossary:
            glossary = ", ".join(self.active_glossary)
            parts.append(f"Active Glossary: {glossary}")
        if self.session_summary:
            parts.append(f"Session Summary: {self.session_summary}")
        return "\n".join(parts)

    def estimate_tokens(self) -> int:
        text = self.to_context_string()
        chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
        other_chars = len(text) - chinese_chars
        return chinese_chars // 2 + other_chars // 4
