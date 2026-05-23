from __future__ import annotations

import json
import re
from datetime import datetime

from scholar_lens.api.schemas import (
    BriefContribution,
    BriefEvidence,
    BriefMethodStep,
    BriefReadingFocus,
    BriefReviewQuestion,
    BriefTerm,
    PaperBriefResponse,
    SectionSummary,
)


TECH_TERM_PATTERNS = [
    "Transformer",
    "self-attention",
    "multi-head attention",
    "attention",
    "embedding",
    "reranker",
    "retrieval",
    "RAG",
    "LLM",
    "CNN",
    "RNN",
    "BERT",
    "GPT",
    "LoRA",
    "softmax",
]


def _section_title(section_id: str, sections: list[SectionSummary]) -> str:
    section = next((s for s in sections if s.section_id == section_id), None)
    return section.title if section else section_id


def _chunk_section_id(chunk: dict) -> str:
    return str(chunk.get("metadata", {}).get("section_id", ""))


def _quote(text: str, limit: int = 220) -> str:
    normalized = " ".join((text or "").split())
    return normalized[:limit]


def _chunks_by_section(chunks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        sid = _chunk_section_id(chunk)
        grouped.setdefault(sid, []).append(chunk)
    return grouped


def _first_chunk_for_section(section_id: str, grouped: dict[str, list[dict]]) -> dict | None:
    items = grouped.get(section_id) or []
    return items[0] if items else None


def _evidence_for_section(section: SectionSummary, grouped: dict[str, list[dict]]) -> BriefEvidence | None:
    chunk = _first_chunk_for_section(section.section_id, grouped)
    text = chunk.get("text", "") if chunk else section.gist
    if not text:
        return None
    return BriefEvidence(section_id=section.section_id, section_title=section.title, quote=_quote(text))


def _extract_terms(text: str) -> list[BriefTerm]:
    found: list[str] = []
    lower = text.lower()
    for term in TECH_TERM_PATTERNS:
        if term.lower() in lower and term not in found:
            found.append(term)
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9-]{2,}\b", text):
        term = match.group(0)
        if term not in found and len(found) < 10:
            found.append(term)
    return [
        BriefTerm(term=term, explanation_zh=f"阅读本文时需要保留英文理解的关键术语：{term}。")
        for term in found[:10]
    ]


def build_fallback_brief(
    doc_id: str,
    title: str,
    sections: list[SectionSummary],
    chunks: list[dict],
) -> PaperBriefResponse:
    grouped = _chunks_by_section(chunks)
    top_sections = [s for s in sections if s.level <= 2] or sections
    all_text = "\n".join(chunk.get("text", "") for chunk in chunks[:12])
    first_section = top_sections[0] if top_sections else None
    method_sections = [
        s for s in top_sections
        if any(word in s.title.lower() for word in ["method", "model", "architecture", "approach", "framework"])
    ] or top_sections[1:3] or top_sections[:1]
    experiment_sections = [
        s for s in top_sections
        if any(word in s.title.lower() for word in ["experiment", "result", "evaluation", "analysis"])
    ]

    tldr = [
        f"本文档《{title}》包含 {len(sections)} 个章节和 {len(chunks)} 个文本片段。",
        f"建议先阅读 {first_section.title if first_section else '开头章节'}，明确论文问题和背景。",
        "方法部分应重点关注模型结构、关键假设以及它相对已有方法的变化。",
    ]
    if experiment_sections:
        tldr.append(f"实验或结果部分可用于判断方法是否真的有效，优先查看 {experiment_sections[0].title}。")

    contributions = []
    for section in (method_sections + experiment_sections + top_sections)[:4]:
        contributions.append(BriefContribution(
            claim=f'围绕“{section.title}”展开论文的一个核心论点或模块。',
            why_it_matters="该部分通常承载论文的新方法、关键实验或主要论证。",
            evidence=_evidence_for_section(section, grouped),
        ))

    if len(method_sections) < 2:
        extra = [s for s in top_sections if s not in method_sections]
        method_sections = (method_sections + extra)[:5]

    method_walkthrough = []
    for index, section in enumerate(method_sections[:5], start=1):
        method_walkthrough.append(BriefMethodStep(
            title=f"Step {index}: {section.title}",
            explanation=section.gist or f"阅读 {section.title}，提取该步骤的输入、处理过程和输出。",
            evidence=_evidence_for_section(section, grouped),
        ))

    reading_focus = [
        BriefReadingFocus(
            section_id=section.section_id,
            section_title=section.title,
            reason="该章节有助于建立论文主线或验证方法有效性。",
        )
        for section in top_sections[:6]
    ]

    review_questions = [
        BriefReviewQuestion(question="这篇论文主要解决什么问题？", level="basic", expected_answer_hint="从 Introduction 或 Abstract 中找问题定义。"),
        BriefReviewQuestion(question="作者提出的方法由哪些关键模块组成？", level="basic", expected_answer_hint="从 Method/Model/Architecture 章节提取模块。"),
        BriefReviewQuestion(question="这个方法相比已有方法的新意是什么？", level="deep", expected_answer_hint="比较 Background 和 Method 中的差异。"),
        BriefReviewQuestion(question="实验是否足以支撑作者的核心结论？", level="critical", expected_answer_hint="检查 Results/Evaluation 的指标和对照。"),
        BriefReviewQuestion(question="如果复现这篇论文，最容易出问题的假设或步骤是什么？", level="critical", expected_answer_hint="关注数据、模型细节和实验设置。"),
    ]

    return PaperBriefResponse(
        doc_id=doc_id,
        title=title,
        source="fallback",
        tldr=tldr[:5],
        problem=_quote(all_text, 300) or "未能从文档中可靠抽取问题定义，请先查看 Introduction。",
        motivation="优先从背景、引言和实验动机中判断该问题为什么值得研究。",
        contributions=contributions[:4],
        method_walkthrough=method_walkthrough[:6],
        key_terms=_extract_terms(all_text),
        reading_focus=reading_focus,
        review_questions=review_questions,
        limitations=["当前为 fallback brief：基于章节和文本片段生成，未调用 LLM 深度归纳。"],
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


# ---- LLM Brief Generation ----

def build_llm_brief_prompt(title: str, sections: list[SectionSummary], chunks: list[dict]) -> str:
    section_lines = "\n".join(
        f"- {s.section_id}: {s.title} (level={s.level}) gist={s.gist[:240]}"
        for s in sections[:20]
    )
    chunk_lines = "\n\n".join(
        f"[chunk {i + 1}] section={_chunk_section_id(chunk)}\n{_quote(chunk.get('text', ''), 700)}"
        for i, chunk in enumerate(chunks[:16])
    )
    return f"""You are generating a Paper Understanding Brief for Chinese university students.

Return ONLY valid JSON. No markdown fences.

The JSON object must contain:
- tldr: 3-5 Chinese sentences
- problem: string
- motivation: string
- contributions: array of 2-4 objects {{claim, why_it_matters, evidence?}}
- method_walkthrough: array of 3-6 objects {{title, explanation, evidence?}}
- key_terms: array of 5-10 objects {{term, explanation_zh, keep_english}}
- reading_focus: array of 3-6 objects {{section_id, section_title, reason}}
- review_questions: array of 5 objects {{question, level, expected_answer_hint}}
- limitations: array of strings

Evidence objects must use actual section ids/titles from the section list and short quotes from chunks.
Preserve English model/method names such as Transformer, self-attention, RAG, BERT, GPT, LoRA.

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


def parse_llm_brief_json(doc_id: str, title: str, raw: str) -> PaperBriefResponse:
    data = _extract_json_object(raw)
    return PaperBriefResponse(
        doc_id=doc_id,
        title=title,
        source="llm",
        tldr=data.get("tldr", []),
        problem=data.get("problem", ""),
        motivation=data.get("motivation", ""),
        contributions=data.get("contributions", []),
        method_walkthrough=data.get("method_walkthrough", []),
        key_terms=data.get("key_terms", []),
        reading_focus=data.get("reading_focus", []),
        review_questions=data.get("review_questions", []),
        limitations=data.get("limitations", []),
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
