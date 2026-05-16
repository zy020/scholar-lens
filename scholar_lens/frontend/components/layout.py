from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MainLayout:
    """Main layout state for the ScholarLens frontend.

    Per spec Section 7.1:
    - Header: ScholarLens title, document selector, mode switch, settings
    - Split view: Document Reader (left) | Interaction Panel (right)
    - Status bar: progress, comprehension, tokens
    """

    current_mode: str = "chat"  # chat | parallel | notes
    current_doc_id: str = ""
    is_configured: bool = False

    _MODES = ("chat", "parallel", "notes")

    def set_mode(self, mode: str) -> None:
        if mode in self._MODES:
            self.current_mode = mode

    def set_document(self, doc_id: str) -> None:
        self.current_doc_id = doc_id
