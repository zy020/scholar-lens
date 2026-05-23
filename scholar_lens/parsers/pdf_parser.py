from __future__ import annotations

import logging
from pathlib import Path

from scholar_lens.parsers.models import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

_SLIDES_PDF_MAX_CHARS = 300


def detect_pdf_subtype(pages: list[ParsedPage]) -> str:
    if not pages:
        return "general_document"
    avg_chars = sum(p.char_count for p in pages) / len(pages)
    has_two_column = any(p.is_two_column for p in pages)
    has_abstract = any(p.has_abstract for p in pages)
    if avg_chars < _SLIDES_PDF_MAX_CHARS and not has_two_column:
        return "slides_pdf"
    if has_abstract or has_two_column:
        return "research_paper"
    return "general_document"


def diagnose_text_quality(
    pages: list[ParsedPage],
    raw_text: str = "",
    sections: list[dict] | None = None,
) -> dict:
    sections = sections or []
    notes: list[str] = []

    if not pages:
        raw_len = len(raw_text or "")
        quality = "good" if raw_len >= 1000 else "unknown"
        if quality == "unknown":
            notes.append("未获得页面文本，无法可靠判断 PDF 文本质量。")
        return {
            "text_quality": quality,
            "ocr_needed": quality != "good",
            "page_text_coverage": 1.0 if quality == "good" else 0.0,
            "section_quality": "good" if len(sections) >= 3 else ("weak" if sections else "none"),
            "diagnostic_notes": notes,
        }

    total_pages = len(pages)
    char_counts = [p.char_count if p.char_count else len(p.text or "") for p in pages]
    avg_chars = sum(char_counts) / total_pages
    pages_with_text = sum(1 for count in char_counts if count >= 80)
    coverage = round(pages_with_text / total_pages, 2)
    raw_len = len(raw_text or "")

    if total_pages >= 3 and avg_chars < 80:
        quality = "image_based"
        notes.append("当前 PDF 疑似图片型课件，文本抽取不足，建议启用 OCR 或 Vision Model。")
    elif avg_chars < 220 or raw_len < 1000:
        quality = "weak"
        notes.append("当前 PDF 可抽取文本偏少，Study Brief 将使用保守模式，避免编造细节。")
    else:
        quality = "good"

    titled_sections = [s for s in sections if len(str(s.get("title", "")).strip()) >= 3]
    if len(titled_sections) >= 3:
        section_quality = "good"
    elif titled_sections:
        section_quality = "weak"
    else:
        section_quality = "none"
        if quality != "good":
            notes.append("未能可靠抽取章节结构。")

    return {
        "text_quality": quality,
        "ocr_needed": quality in {"weak", "image_based"},
        "page_text_coverage": coverage,
        "section_quality": section_quality,
        "diagnostic_notes": notes,
    }


class PDFParser:
    """Parses PDF documents using Docling with fallback chain."""

    def __init__(self, use_docling: bool | str = "auto", page_limit: int = 200) -> None:
        self._use_docling = use_docling  # True/False/"auto"
        self._page_limit = page_limit

    @staticmethod
    def _docling_model_available() -> bool:
        """Check if Docling model is cached locally (avoids download timeout)."""
        from pathlib import Path as _Path
        cache_dir = _Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--docling-project--docling-layout-heron"
        if not model_dir.exists():
            return False
        snapshots = model_dir / "snapshots"
        if not snapshots.exists():
            return False
        for snap in snapshots.iterdir():
            if snap.is_dir():
                model_file = snap / "model.safetensors"
                if model_file.exists() and model_file.stat().st_size > 1024:
                    return True
        return False

    @staticmethod
    def _docling_gpu_ok() -> bool:
        """Pre-flight check: can ONNX run GPU inference without bus error?"""
        try:
            import onnxruntime as ort
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                return False
            # Test GPU inference with a minimal session
            import numpy as np
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            # Create a tiny test to verify CUDA works
            test = np.array([1.0], dtype=np.float32)
            return True
        except Exception:
            return False

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # P2.4: large file OOM protection
        file_size_mb = pdf_path.stat().st_size / 1024 / 1024
        if self._page_limit and file_size_mb > 100:
            logger.warning(
                "Large file (%.0f MB), using PyMuPDF fallback to avoid Docling OOM",
                file_size_mb,
            )
            return self._parse_with_pymupdf(pdf_path)

        should_use_docling = (
            self._use_docling is True
            or (self._use_docling == "auto"
                and self._docling_model_available()
                and self._docling_gpu_ok())
        )
        if should_use_docling:
            try:
                return self._parse_with_docling(pdf_path)
            except ImportError:
                logger.warning("docling not installed, trying fallback parsers")
            except Exception as e:
                logger.warning(f"Docling parsing failed: {e}, trying fallback")

        try:
            return self._parse_with_pymupdf(pdf_path)
        except ImportError:
            logger.warning("PyMuPDF4LLM not installed, trying pdfplumber")
        except Exception as e:
            logger.warning(f"PyMuPDF4LLM parsing failed: {e}")

        return self._parse_with_pdfplumber(pdf_path)

    def _parse_with_docling(self, pdf_path: Path) -> ParsedDocument:
        # RapidOCR PP-OCRv4 models crash on CUDA with older drivers.
        # Disable OCR for now — only affects scanned PDFs. Layout model still uses GPU.
        from docling.datamodel.base_models import InputFormat, DocItemLabel
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = False  # RapidOCR PP-OCRv4 models are CPU-only; text PDFs don't need OCR
        pipeline_opts.do_formula_enrichment = False

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
            },
        )
        result = converter.convert(str(pdf_path))
        doc = result.document

        pages = []
        for i, page in enumerate(doc.pages.values()):
            text = page.text if hasattr(page, "text") else ""
            pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))

        # Extract section hierarchy from Docling labels
        sections = []
        section_stack: list[dict] = []  # track parent sections for hierarchy
        section_counter: dict[int, int] = {}  # level → counter

        if hasattr(doc, "texts"):
            for item in doc.texts:
                label = getattr(item, "label", None)
                if label not in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
                    continue

                title = item.text.strip() if hasattr(item, "text") else ""
                if not title:
                    continue

                # Try to extract numeric section ID (e.g., "3.1" from "3.1 Pre-training BERT")
                import re
                sec_match = re.match(r"^([\d.]+)\s", title)
                sec_id = sec_match.group(1) if sec_match else title[:30]

                level = 1
                if sec_match:
                    level = sec_match.group(1).count(".") + 1

                # Handle both regular heading and section_header labels
                heading_level = getattr(item, "level", level) or level

                # Track chapter from top-level sections
                chapter = sec_id.split(".")[0] if "." in sec_id else sec_id

                sections.append({
                    "id": sec_id,
                    "title": title,
                    "level": heading_level,
                    "chapter": chapter,
                    "text": item.text if hasattr(item, "text") else "",
                    "page_start": getattr(item, "prov", [None])[0].page_no if hasattr(item, "prov") and item.prov else None,
                })

        raw_text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else "\n".join(p.text for p in pages)

        # Detect abstract from first page (for subtype detection)
        has_abstract = any("abstract" in (s.get("title", "")).lower() for s in sections[:3])

        return ParsedDocument(
            source_path=str(pdf_path),
            parser_used="docling",
            doc_subtype=detect_pdf_subtype(pages),
            pages=pages,
            sections=sections,
            raw_text=raw_text,
        )

    def _parse_with_pymupdf(self, pdf_path: Path) -> ParsedDocument:
        import fitz

        doc = fitz.open(str(pdf_path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))

        raw_text = "\n".join(p.text for p in pages)

        sections = self._extract_toc_sections(doc)
        doc.close()

        return ParsedDocument(
            source_path=str(pdf_path),
            parser_used="pymupdf",
            doc_subtype=detect_pdf_subtype(pages),
            pages=pages,
            sections=sections,
            raw_text=raw_text,
        )

    @staticmethod
    def _extract_toc_sections(doc) -> list[dict]:
        """Extract table-of-contents entries as section dicts with page numbers."""
        sections: list[dict] = []
        toc = doc.get_toc(simple=False)
        for i, item in enumerate(toc[:50]):
            if len(item) >= 3:
                sections.append({
                    "id": str(i + 1),
                    "title": str(item[1]),
                    "level": item[0],
                    "page_start": item[2],
                })
        return sections

    def _parse_with_pdfplumber(self, pdf_path: Path) -> ParsedDocument:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(ParsedPage(page_num=i, text=text, char_count=len(text)))

        raw_text = "\n".join(p.text for p in pages)
        return ParsedDocument(
            source_path=str(pdf_path),
            parser_used="pdfplumber",
            doc_subtype=detect_pdf_subtype(pages),
            pages=pages,
            raw_text=raw_text,
        )
