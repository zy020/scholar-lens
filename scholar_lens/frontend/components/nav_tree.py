from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NavTreeState:
    """State for the section navigation tree.

    Per spec Section 7.1: Section navigation with progress indicators.
    """

    sections: list[dict] = field(default_factory=list)  # [{"id": ..., "title": ..., "level": ..., "completed": ...}]
    expanded: set[str] = field(default_factory=set)

    def set_sections(self, sections: list[dict]) -> None:
        self.sections = sections

    def toggle_expand(self, section_id: str) -> None:
        if section_id in self.expanded:
            self.expanded.discard(section_id)
        else:
            self.expanded.add(section_id)
