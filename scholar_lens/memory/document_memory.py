from __future__ import annotations

from scholar_lens.parsers.models import Chunk


class DocumentMemory:
    """Tier 3: Document memory — L0/L1/L2 layered content management.

    L0: Section summaries (~100 token/section)
    L1: Section overviews (~2k token/section)
    L2: Raw chunks (stored in vector store, loaded on demand)
    """

    def __init__(self) -> None:
        self.doc_id: str = ""
        self._l0_summaries: dict[str, str] = {}
        self._l1_overviews: dict[str, str] = {}
        self._l2_chunks: dict[str, list[Chunk]] = {}
        self._mermaid_map: str = ""

    def clear(self) -> None:
        self.doc_id = ""
        self._l0_summaries = {}
        self._l1_overviews = {}
        self._l2_chunks = {}
        self._mermaid_map = ""

    def load_from_document_understanding(
        self,
        l0: dict[str, str],
        l1: dict[str, str],
        l2_chunks: dict[str, list[Chunk]] | None = None,
        mermaid_map: str = "",
        doc_id: str = "",
    ) -> None:
        self.clear()
        self.doc_id = doc_id
        self._l0_summaries = l0
        self._l1_overviews = l1
        if l2_chunks:
            self._l2_chunks = l2_chunks
        self._mermaid_map = mermaid_map

    def get_l0_summary(self, section_id: str) -> str:
        return self._l0_summaries.get(section_id, "")

    def get_l1_overview(self, section_id: str) -> str:
        return self._l1_overviews.get(section_id, self.get_l0_summary(section_id))

    def get_l2_chunks(self, section_id: str) -> list[Chunk]:
        return self._l2_chunks.get(section_id, [])

    def get_mermaid_map(self) -> str:
        return self._mermaid_map

    def get_all_section_ids(self) -> list[str]:
        return list(self._l0_summaries.keys())
