from __future__ import annotations

from scholar_lens.parsers.models import Chunk


class LayeredLoader:
    """L0 → L1 → L2 layered content loading per spec Section 5.1 / 6.1."""

    def load(self, section_id: str, l0_chunks: list[Chunk], l1_chunks: dict[str, Chunk], need_detail: bool = False) -> Chunk | None:
        l0 = self._find_chunk_for_section(l0_chunks, section_id)
        if l0 is None:
            return None
        if not need_detail:
            return l0
        l1 = l1_chunks.get(section_id)
        if l1 is not None:
            return l1
        return l0

    def _find_chunk_for_section(self, chunks: list[Chunk], section_id: str) -> Chunk | None:
        for chunk in chunks:
            if chunk.metadata.section_id == section_id:
                return chunk
        return None
