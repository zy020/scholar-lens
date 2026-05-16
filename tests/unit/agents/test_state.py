import pytest
from scholar_lens.agents.state import ScholarLensState, AgentStep


class TestScholarLensState:
    def test_create_empty(self):
        state = ScholarLensState()
        assert state.doc_id == ""
        assert state.messages == []
        assert state.current_step == ""

    def test_create_with_data(self):
        state = ScholarLensState(
            doc_id="paper_001",
            messages=[{"role": "user", "content": "Explain section 3.1"}],
            current_step="explainer",
        )
        assert state.doc_id == "paper_001"
        assert len(state.messages) == 1

    def test_add_message(self):
        state = ScholarLensState()
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi there")
        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "user"

    def test_agent_step_enum(self):
        assert AgentStep.ANALYZE == "analyze"
        assert AgentStep.EXPLAIN == "explain"
        assert AgentStep.VALIDATE == "validate"
        assert AgentStep.TUTOR == "tutor"
