from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PDFViewerState:
    """State for the PDF.js document viewer component.

    Per spec Section 7.3: Select text, click term, page change,
    click reference, section nav, highlight.
    """

    current_page: int = 0
    total_pages: int = 0
    selected_text: str = ""
    current_section: str = ""
    zoom_level: float = 1.0
