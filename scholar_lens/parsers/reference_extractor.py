from __future__ import annotations

import re
from dataclasses import dataclass, field

from scholar_lens.core.models import CitationContext, Reference


@dataclass
class ReferenceExtraction:
    references: list[Reference] = field(default_factory=list)
    citation_contexts: list[CitationContext] = field(default_factory=list)
    body_text: str = ""


class ReferenceExtractor:
    """Separates reference list from citation contexts in document text."""

    _REF_HEADER = re.compile(r"\n\s*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n", re.MULTILINE)
    _REF_ENTRY = re.compile(r"\[(\d+)\]\s*(.+?)(?=\n\[|\Z)", re.DOTALL)
    _CITATION = re.compile(r"\[(\d+(?:[,\-]\s*\d+)*)\]")

    def extract(self, text: str) -> ReferenceExtraction:
        body_text, ref_section = self._split_references(text)
        references = self._parse_references(ref_section)
        citation_contexts = self._find_citation_contexts(body_text)
        return ReferenceExtraction(
            references=references,
            citation_contexts=citation_contexts,
            body_text=body_text.strip(),
        )

    def _split_references(self, text: str) -> tuple[str, str]:
        match = self._REF_HEADER.search(text)
        if match:
            return text[:match.start()], text[match.end():]
        return text, ""

    def _parse_references(self, ref_section: str) -> list[Reference]:
        refs = []
        for match in self._REF_ENTRY.finditer(ref_section):
            ref_id = match.group(1)
            entry_text = match.group(2).strip()
            authors, title, year, venue = self._parse_entry(entry_text)
            refs.append(Reference(ref_id=ref_id, authors=authors, title=title, year=year, venue=venue))
        return refs

    def _parse_entry(self, text: str) -> tuple[list[str], str, int | None, str | None]:
        year_match = re.search(r"\b((?:19|20)\d{2})\b", text)
        year = int(year_match.group(1)) if year_match else None
        parts = [p.strip() for p in text.split(".") if p.strip()]
        authors = []
        title = ""
        venue = None
        if len(parts) >= 1:
            authors = [a.strip() for a in re.split(r",\s*", parts[0]) if a.strip()]
        if len(parts) >= 2:
            title = parts[1].strip()
        if len(parts) >= 3:
            venue = parts[2].strip()
        return authors, title, year, venue

    def _find_citation_contexts(self, body_text: str) -> list[CitationContext]:
        contexts = []
        for match in self._CITATION.finditer(body_text):
            ref_ids = self._expand_citation_ids(match.group(1))
            start = max(0, match.start() - 200)
            end = min(len(body_text), match.end() + 200)
            surrounding = body_text[start:end].strip()
            for rid in ref_ids:
                contexts.append(CitationContext(ref_id=rid, in_text=match.group(0), surrounding_text=surrounding, section_id=""))
        return contexts

    def _expand_citation_ids(self, ids_str: str) -> list[str]:
        result = []
        for part in re.split(r"[,\s]+", ids_str):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    for i in range(int(start), int(end) + 1):
                        result.append(str(i))
                except ValueError:
                    result.append(part)
            else:
                result.append(part)
        return result
