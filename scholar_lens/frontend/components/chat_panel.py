from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatPanelState:
    """State for the chat tutoring panel.

    Per spec Section 7.2 (Mode 2): Socratic dialogue with tutor agent,
    citation links, quick action buttons.
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    is_streaming: bool = False
    doc_id: str = ""
    section_id: str = ""

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        self.messages = []
