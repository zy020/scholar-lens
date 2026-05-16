from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def create_app():
    """Create the NiceGUI frontend application.

    Per spec Section 7:
    - Three modes: Parallel Reading, Chat Tutoring, Learning Notes
    - PDF.js for document rendering
    - NiceGUI ui.splitter for dual-pane layout
    - SSE for streaming chat responses
    """
    try:
        from nicegui import ui, app
    except ImportError:
        logger.error("nicegui not installed. Install with: pip install nicegui")
        return None

    @ui.page("/")
    async def index():
        # Header
        with ui.header().classes("items-center justify-between"):
            ui.label("ScholarLens").classes("text-h5 font-bold")
            ui.select(
                options=["Chat", "Parallel", "Notes"],
                value="Chat",
                on_change=lambda e: None,  # Mode switch handler
            ).props("dense")
            ui.button(icon="settings", on_click=lambda: None).props("flat round")

        # Main content - splitter
        with ui.splitter(value=50).classes("w-full h-full") as splitter:
            with splitter.before:
                ui.label("Document Viewer").classes("text-caption")
                ui.html('<div id="pdf-viewer" style="height: 100%;"></div>')

            with splitter.after:
                ui.label("Interaction Panel").classes("text-caption")
                ui.html('<div id="interaction-panel" style="height: 100%;"></div>')

        # Status bar
        with ui.footer().classes("items-center"):
            ui.label("Progress: 0/0 | Tokens: 0").classes("text-caption")

    return app
