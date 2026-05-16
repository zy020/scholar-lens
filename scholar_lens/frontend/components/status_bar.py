from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StatusBarState:
    """State for the bottom status bar.

    Per spec Section 7.1: Progress, Comprehension, Token count.
    """

    sections_completed: int = 0
    sections_total: int = 0
    comprehension_score: float = 0.0
    tokens_used: int = 0

    @property
    def progress_text(self) -> str:
        return f"{self.sections_completed}/{self.sections_total} sections"

    @property
    def status_text(self) -> str:
        return f"Progress: {self.progress_text} | Comprehension: {self.comprehension_score:.0%} | Tokens: {self.tokens_used}"
