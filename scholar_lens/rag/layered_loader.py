from __future__ import annotations

from scholar_lens.parsers.models import Chunk


class LayeredLoader:
    """L0 → L1 → L2 layered content loading per spec Section 5.1 / 6.1.

    Resolution order:
    1. L0 summary (~100 tokens/section) — always tried first
    2. L1 overview (~2K tokens/section) — loaded when L0 is insufficient
    3. L2 raw chunks — vector retrieval, used only when L0+L1 not enough
    """

    def __init__(self):
        self._l0: dict[str, str] = {}   # section_id → L0 summary text
        self._l1: dict[str, str] = {}   # section_id → L1 overview text

    def load_document(self, l0_summaries: dict[str, str], l1_overviews: dict[str, str]) -> None:
        """Load L0/L1 summaries from DocumentUnderstanding."""
        self._l0 = l0_summaries
        self._l1 = l1_overviews

    def get_l0(self, section_id: str = "") -> str:
        """Get L0 summaries. If section_id is empty, return all concatenated."""
        if section_id:
            return self._l0.get(section_id, "")
        return "\n\n".join(self._l0.values())

    def get_l1(self, section_id: str) -> str:
        """Get L1 overview for a specific section."""
        return self._l1.get(section_id, "")

    def resolve(
        self, section_id: str = "", need_detail: bool = False,
    ) -> tuple[str, str]:
        """Resolve content at the appropriate layer.

        Returns:
            (content, layer_label) — e.g. ("summary text...", "L0")
            layer_label is "L0", "L1", or "L2".
        """
        # Always try L0 first
        l0 = self.get_l0(section_id)
        if l0 and not need_detail:
            return l0, "L0"

        # Need detail: try L1
        l1 = self.get_l1(section_id)
        if l1:
            return l1, "L1"

        # L1 not available: fall back to L0 content, signal caller to also do L2 retrieval
        if l0:
            return l0, "L0+L2"

        # Nothing cached — caller must do full L2 retrieval
        return "", "L2"

    def _find_chunk_for_section(self, chunks: list[Chunk], section_id: str) -> Chunk | None:
        for chunk in chunks:
            if chunk.metadata.section_id == section_id:
                return chunk
        return None
