from __future__ import annotations

import base64
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import requests
from pydantic import BaseModel, Field

from scholar_lens.core.settings import VisionConfig
from scholar_lens.parsers.ocr_executor import extract_pptx_slide_images, render_pdf_pages


class VisionUnavailableError(RuntimeError):
    pass


class VisionPageEnhancement(BaseModel):
    page: int
    text: str = ""
    visual_type: str = "mixed"
    key_observations: list[str] = Field(default_factory=list)
    formula_summary: str = ""
    table_summary: str = ""
    chart_summary: str = ""
    qa_hint: str = ""
    vision_quality: str = "failed"
    reason: str = ""
    error: str = ""


class VisionEnhancementResult(BaseModel):
    status: str = "completed"
    engine: str = "vision"
    pages: list[VisionPageEnhancement] = Field(default_factory=list)
    error: str = ""


class VisionEnhancementExecutor:
    def __init__(
        self,
        config: VisionConfig,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._post = post or requests.post

    def run(self, source_path: str | Path, pages: list[int]) -> VisionEnhancementResult:
        if not self._config.api_key or not self._config.base_url or not self._config.model:
            raise VisionUnavailableError("Vision model is not configured")
        source = Path(source_path)
        if source.suffix.lower() not in {".pdf", ".pptx"}:
            raise VisionUnavailableError("Vision enhancement currently supports PDF and PPTX sources only")
        if not pages:
            return VisionEnhancementResult(status="skipped", pages=[])

        page_results: list[VisionPageEnhancement] = []
        with tempfile.TemporaryDirectory(prefix="scholar_lens_vision_") as tmp:
            prepared = prepare_vision_images(source, pages, Path(tmp))
            for page in pages:
                image_paths = prepared.get(page, [])
                if not image_paths:
                    reason = "pptx_no_embedded_images" if source.suffix.lower() == ".pptx" else "render_failed"
                    error = (
                        f"Slide {page} has no embedded images for lightweight PPTX Vision"
                        if source.suffix.lower() == ".pptx"
                        else f"Page {page} could not be prepared for Vision"
                    )
                    page_results.append(VisionPageEnhancement(
                        page=page,
                        vision_quality="failed",
                        reason=reason,
                        error=error,
                    ))
                    continue
                try:
                    structured = self._describe_images(page, image_paths)
                    text = structured["text"]
                    quality = "good" if len(text.strip()) >= 20 else "weak"
                    page_results.append(VisionPageEnhancement(
                        page=page,
                        text=text.strip(),
                        visual_type=structured["visual_type"],
                        key_observations=structured["key_observations"],
                        formula_summary=structured["formula_summary"],
                        table_summary=structured["table_summary"],
                        chart_summary=structured["chart_summary"],
                        qa_hint=structured["qa_hint"],
                        vision_quality=quality,
                        reason="vision_text_usable" if text.strip() else "vision_empty",
                    ))
                except Exception as exc:
                    page_results.append(VisionPageEnhancement(
                        page=page,
                        vision_quality="failed",
                        reason="vision_failed",
                        error=str(exc),
                    ))

        status = "completed"
        if page_results and all(page.vision_quality == "failed" for page in page_results):
            status = "failed"
        return VisionEnhancementResult(status=status, pages=page_results)

    def _describe_images(self, page: int, image_paths: list[Path]) -> dict:
        content: list[dict] = [{
            "type": "text",
            "text": (
                "Extract and explain the visible academic content on this page or slide. "
                "Preserve formulas, table values, labels, axes, and important English terms. Return JSON only with keys: "
                "text, visual_type(formula|table|chart|diagram|mixed), key_observations(array), "
                "formula_summary, table_summary, chart_summary, qa_hint. "
                "Use empty strings or arrays for missing fields. Do not invent unsupported derivations."
            ),
        }]
        for image_path in image_paths[:3]:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_path)},
            })
        response = self._post(
            _chat_completions_url(self._config.base_url),
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._config.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 1200,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return parse_vision_structured_content(str(data["choices"][0]["message"]["content"]))
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionUnavailableError("Vision response did not contain message content") from exc


def parse_vision_structured_content(content: str) -> dict:
    text = str(content or "").strip()
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return _structured_fallback(text)
    observations = data.get("key_observations", [])
    if not isinstance(observations, list):
        observations = [str(observations)] if observations else []
    structured = {
        "text": str(data.get("text") or "").strip(),
        "visual_type": _visual_type(str(data.get("visual_type") or "mixed")),
        "key_observations": [str(item).strip() for item in observations if str(item).strip()],
        "formula_summary": str(data.get("formula_summary") or "").strip(),
        "table_summary": str(data.get("table_summary") or "").strip(),
        "chart_summary": str(data.get("chart_summary") or "").strip(),
        "qa_hint": str(data.get("qa_hint") or "").strip(),
    }
    if not _has_meaningful_structured_content(structured):
        return structured
    if not structured["text"]:
        structured["text"] = _structured_text(structured)
    else:
        structured["text"] = _structured_text(structured, leading_text=structured["text"])
    return structured


def _extract_json_object(content: str) -> dict | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    if content.strip().startswith("{"):
        candidates.append(content)
    json_like = re.search(r"(\{.*\})", content, flags=re.DOTALL)
    if json_like:
        candidates.append(json_like.group(1))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _structured_fallback(text: str) -> dict:
    return {
        "text": text,
        "visual_type": "mixed",
        "key_observations": [],
        "formula_summary": "",
        "table_summary": "",
        "chart_summary": "",
        "qa_hint": "",
    }


def _structured_text(structured: dict, leading_text: str = "") -> str:
    lines = []
    if leading_text.strip():
        lines.append(leading_text.strip())
    visual_type = structured.get("visual_type") or "mixed"
    lines.append(f"Visual type: {visual_type}")
    if structured.get("key_observations"):
        lines.append("Key observations:")
        lines.extend(f"- {item}" for item in structured["key_observations"])
    for label, key in [
        ("Formula summary", "formula_summary"),
        ("Table summary", "table_summary"),
        ("Chart summary", "chart_summary"),
        ("QA hint", "qa_hint"),
    ]:
        value = str(structured.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines).strip()


def _has_meaningful_structured_content(structured: dict) -> bool:
    return any([
        bool(str(structured.get("text") or "").strip()),
        bool(structured.get("key_observations")),
        bool(str(structured.get("formula_summary") or "").strip()),
        bool(str(structured.get("table_summary") or "").strip()),
        bool(str(structured.get("chart_summary") or "").strip()),
        bool(str(structured.get("qa_hint") or "").strip()),
    ])


def _visual_type(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in {"formula", "table", "chart", "diagram", "mixed"} else "mixed"


def prepare_vision_images(source_path: Path, pages: list[int], work_dir: Path) -> dict[int, list[Path]]:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return render_pdf_pages(source_path, pages, work_dir)
    if suffix == ".pptx":
        return extract_pptx_slide_images(source_path, pages, work_dir)
    return {}


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"
