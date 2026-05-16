import pytest
from scholar_lens.frontend.components.notes_panel import NotesPanelState


class TestNotesPanelState:
    def test_create(self):
        state = NotesPanelState()
        assert state.terms == []
        assert state.reading_progress == {}

    def test_add_term(self):
        state = NotesPanelState()
        state.add_term("self-attention", "自注意力", "understood")
        assert len(state.terms) == 1
        assert state.terms[0]["english"] == "self-attention"

    def test_update_progress(self):
        state = NotesPanelState()
        state.update_progress("3.1", 0.8)
        assert state.reading_progress["3.1"] == 0.8

    def test_concept_map(self):
        state = NotesPanelState()
        state.concept_map_mermaid = "graph TD\n  A-->B"
        assert "graph TD" in state.concept_map_mermaid
