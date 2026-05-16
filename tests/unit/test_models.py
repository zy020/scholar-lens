import pytest
from scholar_lens.core.models import (
    Section,
    Term,
    Reference,
    CitationContext,
    DocumentUnderstanding,
    ExplanationRequest,
    ExplanationResult,
    ValidationResult,
    StudentProfile,
    ReadingRecord,
)


class TestSection:
    def test_create_section(self):
        s = Section(
            section_id="3.1",
            title="Model Architecture",
            level=2,
            page_start=3,
            page_end=5,
            section_type="method",
            difficulty="advanced",
        )
        assert s.section_id == "3.1"
        assert s.section_type == "method"

    def test_section_defaults(self):
        s = Section(section_id="1", title="Intro", level=1)
        assert s.page_start is None
        assert s.section_type == "prose"
        assert s.difficulty == "intermediate"


class TestTerm:
    def test_create_term(self):
        t = Term(
            english="self-attention",
            chinese="自注意力",
            definition_en="A mechanism relating different positions of a sequence",
            definition_zh="一种关联序列中不同位置的机制",
            relation_type="Used-for",
        )
        assert t.english == "self-attention"
        assert t.chinese == "自注意力"

    def test_term_defaults(self):
        t = Term(english="softmax", chinese="softmax")
        assert t.relation_type is None


class TestReference:
    def test_create_reference(self):
        r = Reference(
            ref_id="1",
            authors=["Vaswani, A.", "Shazeer, N."],
            title="Attention Is All You Need",
            year=2017,
            venue="NeurIPS",
        )
        assert r.ref_id == "1"
        assert len(r.authors) == 2


class TestDocumentUnderstanding:
    def test_create_minimal(self):
        du = DocumentUnderstanding(
            doc_type="research_paper",
            language="en",
            difficulty="advanced",
            estimated_reading_time=45,
            sections=[Section(section_id="1", title="Introduction", level=1)],
            mermaid_map="graph TD\n  A-->B",
            key_terms=[Term(english="transformer", chinese="Transformer")],
            l0_summaries={"1": "Introduces the problem"},
        )
        assert du.doc_type == "research_paper"
        assert len(du.sections) == 1
        assert du.references == []

    def test_l0_l1_keys_match_sections(self):
        du = DocumentUnderstanding(
            doc_type="courseware",
            language="en",
            difficulty="beginner",
            estimated_reading_time=20,
            sections=[
                Section(section_id="1", title="A", level=1),
                Section(section_id="2", title="B", level=1),
            ],
            mermaid_map="",
            key_terms=[],
            l0_summaries={"1": "summary A", "2": "summary B"},
            l1_overviews={"1": "overview A"},
        )
        assert "2" in du.l0_summaries
        assert "2" not in du.l1_overviews


class TestExplanationResult:
    def test_create_result(self):
        er = ExplanationResult(
            original="The self-attention mechanism computes...",
            translation="自注意力机制计算...",
            explanation="自注意力是一种让序列中每个位置都能关注其他位置的机制",
            related_terms=[Term(english="attention", chinese="注意力")],
            difficulty_level="advanced",
            source_section="3.1",
            confidence="high",
        )
        assert er.confidence == "high"

    def test_confidence_values(self):
        for c in ("high", "medium", "low", "unverified"):
            er = ExplanationResult(
                original="x", translation="x", explanation="x",
                confidence=c,
            )
            assert er.confidence == c


class TestValidationResult:
    def test_passed(self):
        vr = ValidationResult(passed=True, confidence="high", issues=[])
        assert vr.passed is True

    def test_failed_with_correction(self):
        vr = ValidationResult(
            passed=False,
            confidence="low",
            issues=["Term 'attention' mistranslated"],
            correction="attention should be 注意力, not 关注",
        )
        assert vr.correction is not None


class TestStudentProfile:
    def test_create_profile(self):
        sp = StudentProfile(
            level="intermediate",
            native_language="zh",
            target_language="en",
            strengths=["linear algebra"],
            weaknesses=["probability theory"],
            total_sessions=5,
        )
        assert sp.level == "intermediate"

    def test_defaults(self):
        sp = StudentProfile()
        assert sp.level == "intermediate"
        assert sp.native_language == "zh"
        assert sp.strengths == []


class TestReadingRecord:
    def test_create_record(self):
        rr = ReadingRecord(
            doc_id="paper_001",
            section_id="3.1",
            comprehension_score=0.8,
        )
        assert rr.comprehension_score == 0.8
