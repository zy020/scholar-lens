import pytest
from scholar_lens.rag.layered_loader import LayeredLoader
from scholar_lens.parsers.models import Chunk, ChunkMetadata


class TestLayeredLoader:
    def test_instantiation(self):
        ll = LayeredLoader()
        assert ll is not None

    def test_load_l0_only(self):
        ll = LayeredLoader()
        l0_chunks = [Chunk(chunk_id="l0_1", text="Section about self-attention", metadata=ChunkMetadata(section_id="3.1", section_type="method"), layer="L0")]
        l1_chunks = {"3.1": Chunk(chunk_id="l1_1", text="Self-attention allows each position..." * 50, metadata=ChunkMetadata(section_id="3.1", section_type="method"), layer="L1")}
        result = ll.load(section_id="3.1", l0_chunks=l0_chunks, l1_chunks=l1_chunks, need_detail=False)
        assert result.layer == "L0"

    def test_load_l1_when_needed(self):
        ll = LayeredLoader()
        l0_chunks = [Chunk(chunk_id="l0_1", text="Brief summary", metadata=ChunkMetadata(section_id="3.1", section_type="method"), layer="L0")]
        l1_chunks = {"3.1": Chunk(chunk_id="l1_1", text="Detailed overview of self-attention mechanism", metadata=ChunkMetadata(section_id="3.1", section_type="method"), layer="L1")}
        result = ll.load(section_id="3.1", l0_chunks=l0_chunks, l1_chunks=l1_chunks, need_detail=True)
        assert result.layer == "L1"

    def test_fallback_to_l0_when_l1_missing(self):
        ll = LayeredLoader()
        l0_chunks = [Chunk(chunk_id="l0_1", text="Summary", metadata=ChunkMetadata(section_id="3.1", section_type="method"), layer="L0")]
        result = ll.load(section_id="3.1", l0_chunks=l0_chunks, l1_chunks={}, need_detail=True)
        assert result.layer == "L0"

    def test_no_chunks_returns_none(self):
        ll = LayeredLoader()
        result = ll.load(section_id="3.1", l0_chunks=[], l1_chunks={}, need_detail=False)
        assert result is None
