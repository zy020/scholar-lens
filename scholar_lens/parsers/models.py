from __future__ import annotations

from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    page_num: int
    text: str
    char_count: int = 0
    is_two_column: bool = False
    has_abstract: bool = False


class ParsedDocument(BaseModel):
    source_path: str
    doc_subtype: str  # research_paper | slides_pdf | courseware_pptx | general_document
    pages: list[ParsedPage] = Field(default_factory=list)
    sections: list[dict] = Field(default_factory=list)
    raw_text: str = ""
    formulas: list[dict] = Field(default_factory=list)
    tables: list[dict] = Field(default_factory=list)
    images: list[dict] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    section_id: str
    section_type: str = "prose"
    chapter: str = ""
    difficulty_score: float = 0.5
    has_formula: bool = False
    formula_ids: list[str] = Field(default_factory=list)
    cross_refs: list[str] = Field(default_factory=list)
    content_type: str = "text"
    caption: str = ""
    referenced_by: list[str] = Field(default_factory=list)
    doc_id: str = ""
    contextual_prefix: str = ""


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    layer: str = "L2"
