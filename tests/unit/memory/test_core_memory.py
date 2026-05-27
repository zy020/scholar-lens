import pytest
from scholar_lens.memory.core_memory import CoreMemory


class TestCoreMemory:
    def test_create_empty(self):
        cm = CoreMemory()
        assert cm.student_profile == ""
        assert cm.current_position == ""
        assert cm.active_glossary == []
        assert cm.session_summary == ""

    def test_create_with_data(self):
        cm = CoreMemory(
            student_profile="Intermediate CS student, strong in math, weak in NLP",
            current_position="paper_001:3.1",
            active_glossary=["self-attention|||自注意力", "positional encoding|||位置编码"],
            session_summary="Reading Transformer paper, discussed attention mechanism.",
        )
        assert "Intermediate" in cm.student_profile
        assert len(cm.active_glossary) == 2

    def test_to_context_string(self):
        cm = CoreMemory(
            student_profile="Intermediate student",
            current_position="paper_001:3.1",
            active_glossary=["attention|||注意力"],
            session_summary="Discussing attention.",
        )
        context = cm.to_context_string()
        assert "Intermediate student" in context
        assert "paper_001:3.1" in context
        assert "attention|||注意力" in context

    def test_token_estimate(self):
        cm = CoreMemory(
            student_profile="x" * 100,
            current_position="doc:1",
            active_glossary=["term:def"],
            session_summary="y" * 200,
        )
        tokens = cm.estimate_tokens()
        assert tokens > 0

    def test_glossary_max_size_via_add(self):
        cm = CoreMemory()
        for i in range(25):
            cm.add_glossary_entry(f"term{i}", f"def{i}")
        assert len(cm.active_glossary) == 20

    def test_update_position(self):
        cm = CoreMemory()
        cm.update_position("paper_001", "4.2")
        assert cm.current_position == "paper_001:4.2"

    def test_add_glossary_entry_no_duplicate(self):
        cm = CoreMemory()
        cm.add_glossary_entry("attention", "注意力")
        cm.add_glossary_entry("attention", "关注")
        assert len(cm.active_glossary) == 1
        assert "attention|||关注" in cm.active_glossary[0]
