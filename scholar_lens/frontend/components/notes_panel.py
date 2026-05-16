from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NotesPanelState:
    """State for the learning notes panel.

    Per spec Section 7.2 (Mode 3): Reading progress overview,
    term glossary with understanding status, concept relationship graph.
    """

    terms: list[dict] = field(default_factory=list)  # [{"english": ..., "chinese": ..., "status": ...}]
    reading_progress: dict[str, float] = field(default_factory=dict)  # section_id → comprehension
    concept_map_mermaid: str = ""
    highlights: list[dict] = field(default_factory=list)

    def add_term(self, english: str, chinese: str, status: str = "new") -> None:
        self.terms.append({"english": english, "chinese": chinese, "status": status})

    def update_progress(self, section_id: str, comprehension: float) -> None:
        self.reading_progress[section_id] = comprehension
