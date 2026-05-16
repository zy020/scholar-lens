import pytest
from scholar_lens.frontend.components.layout import MainLayout


class TestMainLayout:
    def test_instantiation(self):
        layout = MainLayout()
        assert layout is not None
        assert layout.current_mode == "chat"
