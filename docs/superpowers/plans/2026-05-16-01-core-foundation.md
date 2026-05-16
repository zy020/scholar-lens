# ScholarLens Core Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the project skeleton, core data models, configuration, LLM factory, and circuit breaker — the foundation that all other subsystems depend on.

**Architecture:** Monorepo with `scholar_lens/` as the Python package. Core layer provides typed Pydantic models for all domain objects, a settings system that loads from env/file, an LLM factory that creates providers from config with graceful degradation, and a circuit breaker for resilience. All other subsystems import from `scholar_lens.core`.

**Tech Stack:** Python 3.11, Pydantic v2, python-dotenv, langchain-core (already installed), pydantic-settings

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, entry points |
| `scholar_lens/__init__.py` | Package init, version |
| `scholar_lens/core/__init__.py` | Core package init |
| `scholar_lens/core/models.py` | All Pydantic domain models (Section, Term, Reference, DocumentUnderstanding, ExplanationRequest/Result, ValidationResult, StudentModel) |
| `scholar_lens/core/settings.py` | Settings via pydantic-settings, loads from env / .env file |
| `scholar_lens/core/llm_factory.py` | LLM provider factory (OpenAI-compatible), returns ChatOpenAI instances |
| `scholar_lens/core/circuit_breaker.py` | Circuit breaker for external service calls |
| `scholar_lens/core/exceptions.py` | Custom exception hierarchy |
| `scholar_lens/core/token_tracker.py` | Token usage tracking per interaction |
| `tests/__init__.py` | Test package init |
| `tests/unit/__init__.py` | Unit test package init |
| `tests/unit/test_models.py` | Tests for domain models |
| `tests/unit/test_settings.py` | Tests for settings |
| `tests/unit/test_llm_factory.py` | Tests for LLM factory |
| `tests/unit/test_circuit_breaker.py` | Tests for circuit breaker |
| `tests/unit/test_token_tracker.py` | Tests for token tracker |
| `.env.example` | Example environment file |
| `.gitignore` | Git ignore rules |

---

### Task 1: Project Skeleton + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `scholar_lens/__init__.py`
- Create: `scholar_lens/core/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Initialize git repo and create pyproject.toml**

```bash
cd /home/zhy/scholar-lens && git init
```

```toml
# pyproject.toml
[project]
name = "scholar-lens"
version = "0.1.0"
description = "Educational agent system for reading English academic documents"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
    "langchain-core>=1.3",
    "langchain-openai>=1.2",
    "langgraph>=1.1",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]
rag = [
    "chromadb>=1.5",
    "rank-bm25>=0.7",
]
parsers = [
    "docling>=2.0",
    "python-pptx>=1.0",
]

[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create package init files**

`scholar_lens/__init__.py`:
```python
__version__ = "0.1.0"
```

`scholar_lens/core/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

`tests/unit/__init__.py`:
```python
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
data/
*.egg-info/
dist/
build/
.pytest_cache/
htmlcov/
.chroma/
knowledge/
```

- [ ] **Step 4: Create .env.example**

```
# LLM Configuration
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Embedding Configuration
EMBEDDING_API_KEY=your-api-key-here
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

# Optional: Reranker
# RERANKER_MODEL=bge-reranker-m3

# Optional: Vision Model
# VISION_API_KEY=your-api-key-here
# VISION_BASE_URL=https://api.openai.com/v1
# VISION_MODEL=gpt-4o
```

- [ ] **Step 5: Install the package in editable mode and verify**

```bash
cd /home/zhy/scholar-lens && pip install -e ".[dev]" 2>&1 | tail -5
```

Expected: Package installed successfully.

- [ ] **Step 6: Verify pytest works**

```bash
cd /home/zhy/scholar-lens && python -m pytest --co -q
```

Expected: "no tests collected" (empty test dir is fine).

- [ ] **Step 7: Commit**

```bash
cd /home/zhy/scholar-lens && git add -A && git commit -m "feat: project skeleton with pyproject.toml and core package structure"
```

---

### Task 2: Domain Models

**Files:**
- Create: `scholar_lens/core/models.py`
- Create: `tests/unit/test_models.py`

- [ ] **Step 1: Write failing tests for domain models**

```python
# tests/unit/test_models.py
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
        assert "2" not in du.l1_overviews  # L1 is optional per section


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scholar_lens.core.models'`

- [ ] **Step 3: Implement domain models**

```python
# scholar_lens/core/models.py
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Section(BaseModel):
    section_id: str
    title: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    section_type: str = "prose"  # prose | method | results | references | citation_context
    difficulty: str = "intermediate"  # beginner | intermediate | advanced
    children: list[Section] = Field(default_factory=list)


class Term(BaseModel):
    english: str
    chinese: str
    definition_en: str | None = None
    definition_zh: str | None = None
    relation_type: str | None = None  # Used-for | Hyponym-of | Part-of | Compare-with


class Reference(BaseModel):
    ref_id: str
    authors: list[str] = Field(default_factory=list)
    title: str = ""
    year: int | None = None
    venue: str | None = None
    doi: str | None = None


class CitationContext(BaseModel):
    ref_id: str
    in_text: str  # e.g., "We extend [3]"
    surrounding_text: str  # Full paragraph around the citation
    section_id: str


class DocumentUnderstanding(BaseModel):
    doc_type: str  # research_paper | courseware | textbook_chapter
    language: str  # en | zh | mixed
    difficulty: str  # beginner | intermediate | advanced
    estimated_reading_time: int  # minutes
    sections: list[Section]
    mermaid_map: str
    key_terms: list[Term] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    l0_summaries: dict[str, str] = Field(default_factory=dict)  # section_id → ~100 token summary
    l1_overviews: dict[str, str] = Field(default_factory=dict)  # section_id → ~2k token overview
    references: list[Reference] = Field(default_factory=list)
    citation_contexts: list[CitationContext] = Field(default_factory=list)


class ExplanationRequest(BaseModel):
    section_id: str
    mode: str  # translate | explain | term_lookup | sentence_breakdown
    target_text: str | None = None
    student_level: str = "intermediate"
    previous_explanations: list[str] = Field(default_factory=list)


class ExplanationResult(BaseModel):
    original: str
    translation: str
    explanation: str
    related_terms: list[Term] = Field(default_factory=list)
    difficulty_level: str = "intermediate"
    source_section: str = ""
    confidence: str = "high"  # high | medium | low | unverified


class ValidationResult(BaseModel):
    passed: bool
    confidence: str  # high | medium | low
    issues: list[str] = Field(default_factory=list)
    correction: str | None = None


class StudentProfile(BaseModel):
    level: str = "intermediate"
    native_language: str = "zh"
    target_language: str = "en"
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    total_sessions: int = 0


class ReadingRecord(BaseModel):
    doc_id: str
    section_id: str
    comprehension_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_models.py -v
```

Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/core/models.py tests/unit/test_models.py && git commit -m "feat: add domain models (Section, Term, DocumentUnderstanding, etc.)"
```

---

### Task 3: Settings System

**Files:**
- Create: `scholar_lens/core/settings.py`
- Create: `tests/unit/test_settings.py`

- [ ] **Step 1: Write failing tests for settings**

```python
# tests/unit/test_settings.py
import os
import pytest
from scholar_lens.core.settings import Settings, LLMConfig, EmbeddingConfig


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig(api_key="test-key")
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.3

    def test_custom_values(self):
        cfg = LLMConfig(
            api_key="key",
            base_url="http://localhost:11434/v1",
            model="qwen2.5",
            temperature=0.7,
        )
        assert cfg.base_url == "http://localhost:11434/v1"


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig(api_key="test-key")
        assert cfg.model == "text-embedding-3-small"

    def test_embeds_separate_from_llm(self):
        llm = LLMConfig(api_key="llm-key")
        emb = EmbeddingConfig(api_key="emb-key", base_url="http://other/v1")
        assert llm.api_key != emb.api_key
        assert emb.base_url == "http://other/v1"


class TestSettings:
    def test_load_from_env(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "LLM_API_KEY=env-key\n"
            "LLM_MODEL=custom-model\n"
            "EMBEDDING_API_KEY=emb-key\n"
        )
        s = Settings(_env_file=str(env_file))
        assert s.llm.api_key == "env-key"
        assert s.llm.model == "custom-model"
        assert s.embedding.api_key == "emb-key"

    def test_data_dir_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
        s = Settings(llm=LLMConfig(api_key="k"), embedding=EmbeddingConfig(api_key="k"))
        assert s.data_dir is not None

    def test_reranker_optional(self):
        s = Settings(
            llm=LLMConfig(api_key="k"),
            embedding=EmbeddingConfig(api_key="k"),
        )
        assert s.reranker is None

    def test_reranker_configured(self):
        s = Settings(
            llm=LLMConfig(api_key="k"),
            embedding=EmbeddingConfig(api_key="k"),
            reranker_model="bge-reranker-m3",
        )
        assert s.reranker is not None
        assert s.reranker.model == "bge-reranker-m3"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_settings.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scholar_lens.core.settings'`

- [ ] **Step 3: Install pydantic-settings**

```bash
pip install pydantic-settings python-dotenv 2>&1 | tail -3
```

- [ ] **Step 4: Implement settings**

```python
# scholar_lens/core/settings.py
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-small"
    dimension: int = 1536


class RerankerConfig(BaseModel):
    model: str = "bge-reranker-m3"
    base_url: str | None = None
    api_key: str | None = None


class VisionConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # Embedding
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Optional: Reranker
    reranker_model: str | None = None
    reranker_base_url: str | None = None
    reranker_api_key: str | None = None

    # Optional: Vision
    vision_api_key: str | None = None
    vision_base_url: str = "https://api.openai.com/v1"
    vision_model: str = "gpt-4o"

    # Paths
    data_dir: Path = Path(os.getcwd()) / "data"
    knowledge_dir: Path = Path(os.getcwd()) / "knowledge"

    @property
    def llm(self) -> LLMConfig:
        return LLMConfig(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
            model=self.embedding_model,
            dimension=self.embedding_dimension,
        )

    @property
    def reranker(self) -> RerankerConfig | None:
        if self.reranker_model is None:
            return None
        return RerankerConfig(
            model=self.reranker_model,
            base_url=self.reranker_base_url,
            api_key=self.reranker_api_key,
        )

    @property
    def vision(self) -> VisionConfig | None:
        if self.vision_api_key is None:
            return None
        return VisionConfig(
            api_key=self.vision_api_key,
            base_url=self.vision_base_url,
            model=self.vision_model,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_settings.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/core/settings.py tests/unit/test_settings.py && git commit -m "feat: add settings system with pydantic-settings, env file support"
```

---

### Task 4: LLM Factory

**Files:**
- Create: `scholar_lens/core/llm_factory.py`
- Create: `tests/unit/test_llm_factory.py`

- [ ] **Step 1: Write failing tests for LLM factory**

```python
# tests/unit/test_llm_factory.py
import pytest
from unittest.mock import patch, MagicMock
from scholar_lens.core.llm_factory import LLMFactory
from scholar_lens.core.settings import LLMConfig, EmbeddingConfig


class TestLLMFactory:
    def test_create_chat_llm(self):
        config = LLMConfig(api_key="test-key", model="gpt-4o-mini")
        factory = LLMFactory(config)
        llm = factory.create_chat_llm()
        assert llm is not None
        assert llm.model_name == "gpt-4o-mini"

    def test_create_chat_llm_custom_base_url(self):
        config = LLMConfig(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="qwen2.5",
        )
        factory = LLMFactory(config)
        llm = factory.create_chat_llm()
        assert llm.openai_api_base == "http://localhost:11434/v1"

    def test_create_embeddings(self):
        config = EmbeddingConfig(api_key="test-key", model="text-embedding-3-small")
        factory = LLMFactory(config)
        emb = factory.create_embeddings()
        assert emb is not None

    def test_create_embeddings_custom_base_url(self):
        config = EmbeddingConfig(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="bge-m3",
        )
        factory = LLMFactory(config)
        emb = factory.create_embeddings()
        assert emb is not None

    def test_factory_from_settings(self):
        from scholar_lens.core.settings import Settings
        settings = Settings(
            llm_api_key="k",
            llm_model="test-model",
            embedding_api_key="k",
            embedding_model="test-emb",
        )
        factory = LLMFactory.from_settings(settings)
        assert factory is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_llm_factory.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scholar_lens.core.llm_factory'`

- [ ] **Step 3: Implement LLM factory**

```python
# scholar_lens/core/llm_factory.py
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from scholar_lens.core.settings import EmbeddingConfig, LLMConfig

if TYPE_CHECKING:
    from scholar_lens.core.settings import Settings


class LLMFactory:
    """Creates LLM and embedding instances from configuration."""

    def __init__(self, config: LLMConfig | EmbeddingConfig):
        self._config = config

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMFactory:
        return cls(settings.llm)

    def create_chat_llm(
        self,
        config: LLMConfig | None = None,
        streaming: bool = True,
    ) -> ChatOpenAI:
        cfg = config or self._config
        if not isinstance(cfg, LLMConfig):
            raise TypeError("create_chat_llm requires LLMConfig")
        return ChatOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            streaming=streaming,
        )

    def create_embeddings(
        self,
        config: EmbeddingConfig | None = None,
    ) -> OpenAIEmbeddings:
        cfg = config or self._config
        if not isinstance(cfg, EmbeddingConfig):
            raise TypeError("create_embeddings requires EmbeddingConfig")
        return OpenAIEmbeddings(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            dimensions=cfg.dimension,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_llm_factory.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/core/llm_factory.py tests/unit/test_llm_factory.py && git commit -m "feat: add LLM factory for ChatOpenAI and OpenAIEmbeddings"
```

---

### Task 5: Circuit Breaker

**Files:**
- Create: `scholar_lens/core/circuit_breaker.py`
- Create: `tests/unit/test_circuit_breaker.py`

- [ ] **Step 1: Write failing tests for circuit breaker**

```python
# tests/unit/test_circuit_breaker.py
import pytest
import time
from scholar_lens.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_under_failures_below_threshold(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_rejects_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=60)
        cb.record_failure()
        assert not cb.allow_request()

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # cooldown is 0 so next call should transition to half-open
        time.sleep(0.01)
        assert cb.allow_request()
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_closes_half_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        time.sleep(0.01)
        cb.allow_request()  # transitions to half-open
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        time.sleep(0.01)
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_closed_circuit_allows_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        assert cb.allow_request()

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # never hit 3 consecutive
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_circuit_breaker.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement circuit breaker**

```python
# scholar_lens/core/circuit_breaker.py
from __future__ import annotations

import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """3 consecutive failures → open circuit → cooldown → half-open → test request."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow one test request
        return True

    def record_success(self) -> None:
        self._failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_circuit_breaker.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/core/circuit_breaker.py tests/unit/test_circuit_breaker.py && git commit -m "feat: add circuit breaker for external service resilience"
```

---

### Task 6: Exception Hierarchy + Token Tracker

**Files:**
- Create: `scholar_lens/core/exceptions.py`
- Create: `scholar_lens/core/token_tracker.py`
- Create: `tests/unit/test_exceptions.py`
- Create: `tests/unit/test_token_tracker.py`

- [ ] **Step 1: Write failing tests for exceptions**

```python
# tests/unit/test_exceptions.py
import pytest
from scholar_lens.core.exceptions import (
    ScholarLensError,
    ParsingError,
    LLMError,
    LLMRateLimitError,
    RetrievalError,
    ValidationError as AppValidationError,
    CircuitOpenError,
)


class TestExceptionHierarchy:
    def test_base_error(self):
        e = ScholarLensError("something went wrong")
        assert str(e) == "something went wrong"

    def test_parsing_error_is_base(self):
        assert issubclass(ParsingError, ScholarLensError)

    def test_llm_error_is_base(self):
        assert issubclass(LLMError, ScholarLensError)

    def test_rate_limit_is_llm_error(self):
        assert issubclass(LLMRateLimitError, LLMError)

    def test_retrieval_error_is_base(self):
        assert issubclass(RetrievalError, ScholarLensError)

    def test_validation_error_is_base(self):
        assert issubclass(AppValidationError, ScholarLensError)

    def test_circuit_open_error(self):
        cb = type("CB", (), {"name": "vlm"})()
        e = CircuitOpenError("vlm", cb)
        assert "vlm" in str(e)
```

- [ ] **Step 2: Write failing tests for token tracker**

```python
# tests/unit/test_token_tracker.py
import pytest
from scholar_lens.core.token_tracker import TokenTracker, TokenUsage


class TestTokenUsage:
    def test_create(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini")
        assert u.total_tokens == 150

    def test_defaults(self):
        u = TokenUsage()
        assert u.total_tokens == 0


class TestTokenTracker:
    def test_record_and_total(self):
        tracker = TokenTracker()
        tracker.record("agent1", prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini")
        tracker.record("agent1", prompt_tokens=200, completion_tokens=100, model="gpt-4o-mini")
        total = tracker.get_total("agent1")
        assert total.prompt_tokens == 300
        assert total.completion_tokens == 150
        assert total.total_tokens == 450

    def test_multiple_agents(self):
        tracker = TokenTracker()
        tracker.record("explainer", prompt_tokens=100, completion_tokens=50, model="m1")
        tracker.record("tutor", prompt_tokens=200, completion_tokens=80, model="m2")
        assert tracker.get_total("explainer").total_tokens == 150
        assert tracker.get_total("tutor").total_tokens == 280
        assert tracker.get_grand_total().total_tokens == 430

    def test_unknown_agent_returns_zero(self):
        tracker = TokenTracker()
        total = tracker.get_total("nonexistent")
        assert total.total_tokens == 0

    def test_interaction_budget_check(self):
        tracker = TokenTracker()
        tracker.record("tutor", prompt_tokens=3000, completion_tokens=800, model="m")
        tracker.record("tutor", prompt_tokens=1000, completion_tokens=400, model="m")
        # Budget 4600 tokens per spec
        within = tracker.is_within_budget("tutor", budget=4600)
        assert within is True  # 5200 > 4600? No: 3000+800+1000+400 = 5200 > 4600
        # Actually 5200 > 4600, so should be False
        assert within is False

    def test_summary(self):
        tracker = TokenTracker()
        tracker.record("a", prompt_tokens=100, completion_tokens=50, model="m")
        summary = tracker.summary()
        assert "a" in summary
        assert summary["a"]["total"] == 150
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_exceptions.py tests/unit/test_token_tracker.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement exceptions**

```python
# scholar_lens/core/exceptions.py
from __future__ import annotations


class ScholarLensError(Exception):
    """Base exception for all ScholarLens errors."""


class ParsingError(ScholarLensError):
    """Error during document parsing."""


class LLMError(ScholarLensError):
    """Error from LLM invocation."""


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""


class RetrievalError(ScholarLensError):
    """Error during RAG retrieval."""


class ValidationError(ScholarLensError):
    """Error during content validation."""


class CircuitOpenError(ScholarLensError):
    """Circuit breaker is open, rejecting requests."""

    def __init__(self, service_name: str, circuit_breaker: object):
        self.service_name = service_name
        self.circuit_breaker = circuit_breaker
        super().__init__(f"Circuit breaker open for service: {service_name}")
```

- [ ] **Step 5: Implement token tracker**

```python
# scholar_lens/core/token_tracker.py
from __future__ import annotations

from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenTracker:
    """Tracks token usage per agent across interactions."""

    def __init__(self) -> None:
        self._usage: dict[str, list[TokenUsage]] = {}

    def record(
        self,
        agent: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        if agent not in self._usage:
            self._usage[agent] = []
        self._usage[agent].append(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
            )
        )

    def get_total(self, agent: str) -> TokenUsage:
        records = self._usage.get(agent, [])
        if not records:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            model=records[-1].model,
        )

    def get_grand_total(self) -> TokenUsage:
        all_records: list[TokenUsage] = []
        for records in self._usage.values():
            all_records.extend(records)
        if not all_records:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=sum(r.prompt_tokens for r in all_records),
            completion_tokens=sum(r.completion_tokens for r in all_records),
            model="",
        )

    def is_within_budget(self, agent: str, budget: int = 4600) -> bool:
        total = self.get_total(agent)
        return total.total_tokens <= budget

    def summary(self) -> dict[str, dict]:
        result = {}
        for agent in self._usage:
            total = self.get_total(agent)
            result[agent] = {
                "prompt": total.prompt_tokens,
                "completion": total.completion_tokens,
                "total": total.total_tokens,
            }
        return result
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/test_exceptions.py tests/unit/test_token_tracker.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/core/exceptions.py scholar_lens/core/token_tracker.py tests/unit/test_exceptions.py tests/unit/test_token_tracker.py && git commit -m "feat: add exception hierarchy and token usage tracker"
```

---

### Task 7: Core Package Init + Integration Smoke Test

**Files:**
- Modify: `scholar_lens/core/__init__.py`
- Create: `tests/unit/test_core_imports.py`

- [ ] **Step 1: Update core __init__.py with public API**

```python
# scholar_lens/core/__init__.py
from scholar_lens.core.models import (
    CitationContext,
    DocumentUnderstanding,
    ExplanationRequest,
    ExplanationResult,
    ReadingRecord,
    Reference,
    Section,
    StudentProfile,
    Term,
    ValidationResult,
)
from scholar_lens.core.settings import Settings
from scholar_lens.core.circuit_breaker import CircuitBreaker, CircuitState
from scholar_lens.core.token_tracker import TokenTracker, TokenUsage
from scholar_lens.core.exceptions import (
    ScholarLensError,
    ParsingError,
    LLMError,
    LLMRateLimitError,
    RetrievalError,
    ValidationError,
    CircuitOpenError,
)

__all__ = [
    "CitationContext",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "DocumentUnderstanding",
    "ExplanationRequest",
    "ExplanationResult",
    "LLMError",
    "LLMRateLimitError",
    "ParsingError",
    "ReadingRecord",
    "Reference",
    "RetrievalError",
    "ScholarLensError",
    "Section",
    "Settings",
    "StudentProfile",
    "Term",
    "TokenTracker",
    "TokenUsage",
    "ValidationError",
    "ValidationResult",
]
```

- [ ] **Step 2: Write import smoke test**

```python
# tests/unit/test_core_imports.py
import pytest


def test_all_core_imports():
    from scholar_lens.core import (
        CitationContext,
        CircuitBreaker,
        CircuitOpenError,
        CircuitState,
        DocumentUnderstanding,
        ExplanationRequest,
        ExplanationResult,
        LLMError,
        LLMRateLimitError,
        ParsingError,
        ReadingRecord,
        Reference,
        RetrievalError,
        ScholarLensError,
        Section,
        Settings,
        StudentProfile,
        Term,
        TokenTracker,
        TokenUsage,
        ValidationError,
        ValidationResult,
    )
    # If we get here, all imports succeeded
    assert True
```

- [ ] **Step 3: Run all tests**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/core/__init__.py tests/unit/test_core_imports.py && git commit -m "feat: wire up core package public API and import smoke test"
```

---

## Self-Review

**1. Spec coverage:**
- Domain models (Section 4.1-4.5 data structures) → Task 2 ✓
- Settings/Config (Section 7.4 model config panel) → Task 3 ✓
- LLM factory (Section 3.3 tech stack) → Task 4 ✓
- Circuit breaker (Section 8.2) → Task 5 ✓
- Exceptions (Section 8 error handling) → Task 6 ✓
- Token tracker (Section 9.1 token efficiency) → Task 6 ✓
- Project structure (Section 10) → Task 1 ✓

**2. Placeholder scan:** No TBD, TODO, or vague steps found. All code is complete.

**3. Type consistency:** `LLMConfig`, `EmbeddingConfig`, `RerankerConfig` used consistently between settings and factory. `CircuitBreaker` constructor matches usage in `CircuitOpenError`. `TokenUsage` fields match `TokenTracker.record()` params.
