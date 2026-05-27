from __future__ import annotations

import json
from datetime import datetime

from scholar_lens.api.schemas import (
    PaperBriefResponse,
    SectionSummary,
)


def _chunk_section_id(chunk: dict) -> str:
    return str(chunk.get("metadata", {}).get("section_id", ""))


def _quote(text: str, limit: int = 220) -> str:
    normalized = " ".join((text or "").split())
    return normalized[:limit]


def build_unavailable_brief(
    doc_id: str,
    title: str,
    *,
    brief_type: str = "paper",
    text_quality: str = "unknown",
    ocr_needed: bool = False,
    error: str = "",
) -> PaperBriefResponse:
    reason = error or "文档学习分析需要配置并成功调用 LLM。"
    return PaperBriefResponse(
        doc_id=doc_id,
        title=title,
        source="unavailable",
        brief_type=brief_type,
        text_quality=text_quality,
        ocr_needed=ocr_needed,
        problem="需要配置并成功调用 LLM 后才能生成文档学习分析。",
        motivation="系统不会使用规则摘要兜底，避免用浅层摘要误导学习。",
        limitations=[
            "未生成文档学习分析。",
            "请在配置面板启用可用 LLM 后重新生成文档学习分析。",
        ],
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        error=reason,
    )


def build_not_generated_brief(
    doc_id: str,
    title: str,
    *,
    brief_type: str = "paper",
    text_quality: str = "unknown",
    ocr_needed: bool = False,
) -> PaperBriefResponse:
    return PaperBriefResponse(
        doc_id=doc_id,
        title=title,
        source="not_generated",
        brief_type=brief_type,
        text_quality=text_quality,
        ocr_needed=ocr_needed,
        problem="文档学习分析尚未生成。",
        motivation="点击“生成文档学习分析”后，系统会使用已配置的 LLM 基于当前文档生成结构化学习材料。",
        limitations=["默认不会自动生成文档学习分析，避免上传后立即消耗模型调用。"],
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


# ---- LLM Brief Generation ----

BRIEF_PROMPT_SECTION_LIMIT = 12
BRIEF_PROMPT_CHUNK_LIMIT = 8
BRIEF_PROMPT_GIST_CHARS = 120
BRIEF_PROMPT_CHUNK_CHARS = 420

def build_llm_brief_prompt(title: str, sections: list[SectionSummary], chunks: list[dict]) -> str:
    section_lines = "\n".join(
        f"- {s.section_id}: {s.title} (level={s.level}) gist={s.gist[:BRIEF_PROMPT_GIST_CHARS]}"
        for s in sections[:BRIEF_PROMPT_SECTION_LIMIT]
    )
    chunk_lines = "\n\n".join(
        f"[chunk {i + 1}] section={_chunk_section_id(chunk)}\n{_quote(chunk.get('text', ''), BRIEF_PROMPT_CHUNK_CHARS)}"
        for i, chunk in enumerate(chunks[:BRIEF_PROMPT_CHUNK_LIMIT])
    )
    return f"""You are generating a Paper Understanding Brief for Chinese university students.

Return ONLY valid JSON. No markdown fences.

The JSON object must contain:
- tldr: 3-5 Chinese sentences
- problem: string
- motivation: string
- contributions: array of 2-4 objects {{claim, why_it_matters, evidence?}}
- method_walkthrough: array of 3-6 objects {{title, explanation, evidence?}}
- reading_focus: array of 3-6 objects {{section_id, section_title, reason}}. For slides, use section_id like "slide_3" and section_title like "Slide 4: topic".
- review_questions: array of 5 objects {{question, level, expected_answer_hint}}
- limitations: array of strings

Evidence objects must use actual section ids/titles from the section list and short quotes from chunks.
Preserve English model/method names such as Transformer, self-attention, RAG, BERT, GPT, LoRA.
Do not include a generic glossary or difficulty estimate.

Title: {title}

Sections:
{section_lines}

Representative chunks:
{chunk_lines}
"""


def _extract_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(text[start:end + 1])


def _normalize_evidence(value) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        return {"quote": value.strip()[:260]}
    return None


def _normalize_evidence_items(items) -> list[dict]:
    if not isinstance(items, list):
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        next_item = dict(item)
        if "evidence" in next_item:
            evidence = _normalize_evidence(next_item.get("evidence"))
            if evidence is None:
                next_item.pop("evidence", None)
            else:
                next_item["evidence"] = evidence
        normalized.append(next_item)
    return normalized


def parse_llm_brief_json(doc_id: str, title: str, raw: str) -> PaperBriefResponse:
    data = _extract_json_object(raw)
    contributions = data.get("contributions")
    if contributions is None and data.get("brief_type") == "lecture":
        contributions = data.get("core_concepts", [])
    return PaperBriefResponse(
        doc_id=doc_id,
        title=title,
        source="llm",
        brief_type=data.get("brief_type", "paper"),
        text_quality=data.get("text_quality", "good"),
        ocr_needed=bool(data.get("ocr_needed", False)),
        tldr=data.get("tldr", []),
        problem=data.get("problem", ""),
        motivation=data.get("motivation", ""),
        contributions=_normalize_evidence_items(contributions),
        method_walkthrough=_normalize_evidence_items(data.get("method_walkthrough", [])),
        key_terms=[],
        reading_focus=data.get("reading_focus", []),
        review_questions=data.get("review_questions", []),
        limitations=data.get("limitations", []),
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


def build_lecture_llm_brief_prompt(title: str, sections: list[SectionSummary], chunks: list[dict]) -> str:
    section_lines = "\n".join(
        f"- {s.section_id}: {s.title} (level={s.level}) gist={s.gist[:BRIEF_PROMPT_GIST_CHARS]}"
        for s in sections[:BRIEF_PROMPT_SECTION_LIMIT]
    )
    chunk_lines = "\n\n".join(
        f"[chunk {i + 1}] section={_chunk_section_id(chunk)}\n{_quote(chunk.get('text', ''), BRIEF_PROMPT_CHUNK_CHARS)}"
        for i, chunk in enumerate(chunks[:BRIEF_PROMPT_CHUNK_LIMIT])
    )
    return f"""You are generating a Lecture Learning Analysis for Chinese university students.

Return ONLY valid JSON. No markdown fences.

The JSON object must contain:
- brief_type: "lecture"
- tldr: 3-5 Chinese sentences
- problem: string describing the lecture topic, not a research problem
- motivation: string explaining why the topic matters for learning
- core_concepts: array of 2-5 objects {{claim, why_it_matters, evidence?}} where claim is a core concept or knowledge point
- method_walkthrough: array of 3-6 objects {{title, explanation, evidence?}} representing a learning path
- reading_focus: array of 3-6 objects {{section_id, section_title, reason}}
- review_questions: array of 5 objects {{question, level, expected_answer_hint}}
- limitations: array of strings

Lecture-specific requirements:
- important_slides: reason about which Slide/section ids deserve close review; mirror them into reading_focus with visible Slide page numbers.
- formulas_or_figures: identify formula/figure/diagram slides and explain what students should understand.
- confusion_points: identify concepts that are easy to mix up.
- self-check questions: review_questions must be exam- or self-check-oriented, not paper-review questions.
- Do not include a generic glossary, difficulty estimate, or reading-time estimate.

Preserve English model/method names and symbols. Do not invent content not supported by chunks.

Title: {title}

Sections:
{section_lines}

Representative chunks:
{chunk_lines}
"""
