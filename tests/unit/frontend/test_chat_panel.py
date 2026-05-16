import pytest
from scholar_lens.frontend.components.chat_panel import ChatPanelState


class TestChatPanelState:
    def test_create(self):
        panel = ChatPanelState()
        assert panel.messages == []
        assert panel.is_streaming is False

    def test_add_message(self):
        panel = ChatPanelState()
        panel.add_message("user", "What is self-attention?")
        panel.add_message("assistant", "Self-attention is a mechanism...")
        assert len(panel.messages) == 2
        assert panel.messages[0]["role"] == "user"

    def test_clear_messages(self):
        panel = ChatPanelState()
        panel.add_message("user", "Hello")
        panel.clear()
        assert panel.messages == []

    def test_streaming_state(self):
        panel = ChatPanelState()
        panel.is_streaming = True
        assert panel.is_streaming is True
