import pytest
from scholar_lens.rag.layered_loader import LayeredLoader


class TestLayeredLoader:
    def test_instantiation(self):
        ll = LayeredLoader()
        assert ll is not None

    def test_load_l0_only(self):
        ll = LayeredLoader()
        ll.load_document(
            l0_summaries={"3.1": "Section about self-attention"},
            l1_overviews={"3.1": "Self-attention allows each position..." * 10},
        )
        content, layer = ll.resolve(section_id="3.1", need_detail=False)
        assert layer == "L0"
        assert "self-attention" in content

    def test_load_l1_when_needed(self):
        ll = LayeredLoader()
        ll.load_document(
            l0_summaries={"3.1": "Brief summary"},
            l1_overviews={"3.1": "Detailed overview of self-attention mechanism"},
        )
        content, layer = ll.resolve(section_id="3.1", need_detail=True)
        assert layer == "L1"
        assert "Detailed" in content

    def test_fallback_to_l0_when_l1_missing(self):
        ll = LayeredLoader()
        ll.load_document(
            l0_summaries={"3.1": "Summary"},
            l1_overviews={},
        )
        content, layer = ll.resolve(section_id="3.1", need_detail=True)
        assert layer == "L0+L2"  # L1 missing, falls back to L0 with L2 hint

    def test_no_content_returns_l2(self):
        ll = LayeredLoader()
        content, layer = ll.resolve(section_id="3.1", need_detail=False)
        assert layer == "L2"
        assert content == ""

    def test_get_l0_all_concatenated(self):
        ll = LayeredLoader()
        ll.load_document(
            l0_summaries={"1": "Intro summary", "2": "Method summary"},
            l1_overviews={},
        )
        all_l0 = ll.get_l0()  # no section_id → all concatenated
        assert "Intro summary" in all_l0
        assert "Method summary" in all_l0
