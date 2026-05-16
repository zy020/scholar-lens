import pytest
from scholar_lens.frontend.components.parallel_reader import ParallelReaderState


class TestParallelReaderState:
    def test_create(self):
        state = ParallelReaderState()
        assert state.current_section_id == ""
        assert state.paragraphs == []

    def test_set_paragraphs(self):
        state = ParallelReaderState()
        state.set_paragraphs([
            {"en": "The self-attention mechanism computes attention scores.", "zh": "自注意力机制计算注意力分数。"},
            {"en": "Multi-head attention runs multiple attention functions.", "zh": "多头注意力运行多个注意力函数。"},
        ])
        assert len(state.paragraphs) == 2

    def test_scroll_sync_position(self):
        state = ParallelReaderState()
        state.current_paragraph_index = 3
        assert state.current_paragraph_index == 3
