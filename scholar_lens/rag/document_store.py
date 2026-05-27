"""DocumentStore: structured per-document file storage.

Replaces flat data/uploads/ with data/documents/{doc_id}/ layout.

Each document directory contains:
  metadata.json  - DocumentSummary
  sections.json  - list of SectionSummary
  chunks.jsonl   - one Chunk per line (JSON)
  source.*       - original uploaded file
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from scholar_lens.api.schemas import DocumentStatus, DocumentSummary, SectionSummary

if TYPE_CHECKING:
    from scholar_lens.parsers.models import Chunk, ParsedDocument


class DocumentStore:
    def __init__(self, root: Path | str = "data/documents") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # ---- factory ----

    def create_document(self, filename: str, suffix: str = ".pdf") -> DocumentSummary:
        doc_id = uuid.uuid4().hex[:8]
        doc_dir = self._root / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        name = Path(filename).name or f"document{suffix}"
        summary = DocumentSummary(
            doc_id=doc_id, name=name, status=DocumentStatus.uploaded,
            file_url=f"/api/documents/{doc_id}/file",
        )
        self._write_metadata(doc_id, summary)
        return summary

    # ---- paths ----

    def document_dir(self, doc_id: str) -> Path:
        return self._root / doc_id

    def source_path(self, doc_id: str, suffix: str | None = None) -> Path:
        doc_dir = self.document_dir(doc_id)
        if suffix is not None:
            return doc_dir / f"source{suffix}"

        pdf_path = doc_dir / "source.pdf"
        if pdf_path.exists():
            return pdf_path

        sources = sorted(doc_dir.glob("source.*"))
        if sources:
            return sources[0]
        return pdf_path

    def _metadata_path(self, doc_id: str) -> Path:
        return self.document_dir(doc_id) / "metadata.json"

    def _sections_path(self, doc_id: str) -> Path:
        return self.document_dir(doc_id) / "sections.json"

    def _chunks_path(self, doc_id: str) -> Path:
        return self.document_dir(doc_id) / "chunks.jsonl"

    def _understanding_path(self, doc_id: str) -> Path:
        return self.document_dir(doc_id) / "document_understanding.json"

    def _analysis_meta_path(self, doc_id: str) -> Path:
        return self.document_dir(doc_id) / "document_analysis_meta.json"

    def _parse_quality_path(self, doc_id: str) -> Path:
        return self.document_dir(doc_id) / "parse_quality.json"

    def _ocr_enhancement_path(self, doc_id: str) -> Path:
        return self.document_dir(doc_id) / "ocr_enhancement.json"

    def _vision_enhancement_path(self, doc_id: str) -> Path:
        return self.document_dir(doc_id) / "vision_enhancement.json"

    def _parsed_document_path(self, doc_id: str, enhanced: bool = False) -> Path:
        filename = "parsed_document.enhanced.json" if enhanced else "parsed_document.json"
        return self.document_dir(doc_id) / filename

    # ---- source file ----

    def save_source(self, doc_id: str, content: bytes, suffix: str = ".pdf") -> Path:
        fp = self.document_dir(doc_id) / f"source{suffix}"
        fp.write_bytes(content)
        return fp

    # ---- metadata ----

    def _read_metadata(self, doc_id: str) -> DocumentSummary | None:
        path = self._metadata_path(doc_id)
        if not path.exists():
            return None
        return DocumentSummary(**json.loads(path.read_text(encoding="utf-8")))

    def _write_metadata(self, doc_id: str, summary: DocumentSummary) -> None:
        path = self._metadata_path(doc_id)
        path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    # ---- public ----

    def get(self, doc_id: str) -> DocumentSummary | None:
        return self._read_metadata(doc_id)

    def list(self) -> list[DocumentSummary]:
        docs = []
        for d in sorted(self._root.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                summary = self._read_metadata(d.name)
                if summary:
                    docs.append(summary)
        return docs

    def name_exists(self, filename: str) -> bool:
        target = Path(filename).name
        return any(doc.name == target for doc in self.list())

    def update_status(self, doc_id: str, status: DocumentStatus, error: str = "") -> None:
        summary = self._read_metadata(doc_id)
        if summary is None:
            return
        summary.status = status
        if error:
            summary.error = error
        self._write_metadata(doc_id, summary)

    def update_summary(self, doc_id: str, **fields) -> None:
        summary = self._read_metadata(doc_id)
        if summary is None:
            return
        for key, value in fields.items():
            if hasattr(summary, key):
                setattr(summary, key, value)
        self._write_metadata(doc_id, summary)

    def save_sections(self, doc_id: str, sections: list[SectionSummary]) -> None:
        data = [s.model_dump() for s in sections]
        self._sections_path(doc_id).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = self._read_metadata(doc_id)
        if summary:
            summary.num_sections = len(sections)
            self._write_metadata(doc_id, summary)

    def load_sections(self, doc_id: str) -> list[SectionSummary]:
        path = self._sections_path(doc_id)
        if not path.exists():
            return []
        return [SectionSummary(**s) for s in json.loads(path.read_text(encoding="utf-8"))]

    def save_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        path = self._chunks_path(doc_id)
        with open(path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
        summary = self._read_metadata(doc_id)
        if summary:
            summary.num_chunks = len(chunks)
            self._write_metadata(doc_id, summary)

    def load_chunks(self, doc_id: str) -> list[dict]:
        path = self._chunks_path(doc_id)
        if not path.exists():
            return []
        chunks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def save_understanding(self, doc_id: str, understanding) -> None:
        self._understanding_path(doc_id).write_text(
            understanding.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def load_understanding(self, doc_id: str):
        path = self._understanding_path(doc_id)
        if not path.exists():
            return None
        from scholar_lens.core.models import DocumentUnderstanding
        return DocumentUnderstanding(**json.loads(path.read_text(encoding="utf-8")))

    def save_analysis_meta(self, doc_id: str, meta: dict) -> None:
        self._analysis_meta_path(doc_id).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_analysis_meta(self, doc_id: str) -> dict:
        path = self._analysis_meta_path(doc_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_parse_quality(self, doc_id: str, qualities) -> None:
        data = [q.model_dump() if hasattr(q, "model_dump") else q for q in qualities]
        self._parse_quality_path(doc_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_parse_quality(self, doc_id: str) -> list[dict]:
        path = self._parse_quality_path(doc_id)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def save_ocr_enhancement(self, doc_id: str, payload) -> None:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload
        self._ocr_enhancement_path(doc_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_ocr_enhancement(self, doc_id: str) -> dict:
        path = self._ocr_enhancement_path(doc_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_vision_enhancement(self, doc_id: str, payload) -> None:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload
        self._vision_enhancement_path(doc_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_vision_enhancement(self, doc_id: str) -> dict:
        path = self._vision_enhancement_path(doc_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_parsed_document(self, doc_id: str, parsed: ParsedDocument, enhanced: bool = False) -> None:
        data = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
        self._parsed_document_path(doc_id, enhanced=enhanced).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_parsed_document(self, doc_id: str, enhanced: bool = False):
        path = self._parsed_document_path(doc_id, enhanced=enhanced)
        if not path.exists():
            return None
        from scholar_lens.parsers.models import ParsedDocument
        return ParsedDocument(**json.loads(path.read_text(encoding="utf-8")))

    def delete(self, doc_id: str) -> None:
        doc_dir = self.document_dir(doc_id)
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
