from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParallelReaderState:
    """State for the bilingual parallel reading panel.

    Per spec Section 7.2 (Mode 1): English original + Chinese translation
    side by side, scroll-synced, terms highlighted, hover for details.
    """

    current_section_id: str = ""
    paragraphs: list[dict[str, str]] = field(default_factory=list)  # [{"en": ..., "zh": ...}]
    current_paragraph_index: int = 0
    highlight_terms: list[str] = field(default_factory=list)

    def set_paragraphs(self, paragraphs: list[dict[str, str]]) -> None:
        self.paragraphs = paragraphs
        self.current_paragraph_index = 0
