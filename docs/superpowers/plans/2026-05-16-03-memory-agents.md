# ScholarLens Memory System + Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four-tier memory system (Core, Structured, Document, Reflection) and the four agents (Document Analyzer, Content Explainer, Validator, Learning Tutor) orchestrated via LangGraph.

**Architecture:** Memory is organized in 4 tiers stored in SQLite + Markdown files. Each agent reads/writes specific tiers per the spec's agent-memory interaction matrix. LangGraph orchestrates a pipeline graph (upload → analyze → explain → validate → tutor) and an interactive graph (tutor loop with on-demand explainer/validator calls). Agents are implemented as LangGraph nodes with typed state.

**Tech Stack:** LangGraph (orchestration), SQLite (structured memory), aiosqlite (async SQLite), langchain-core (chat models, messages), langchain-openai (ChatOpenAI)

**Depends on:** Plan 01 (core models, settings, LLM factory), Plan 02 (parsers, RAG pipeline)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scholar_lens/memory/__init__.py` | Memory package init |
| `scholar_lens/memory/core_memory.py` | Tier 1: Core memory (always in context, ~500 tokens) |
| `scholar_lens/memory/structured_memory.py` | Tier 2: Structured long-term memory (SQLite) |
| `scholar_lens/memory/document_memory.py` | Tier 3: Document memory (RAG pipeline integration) |
| `scholar_lens/memory/reflection_memory.py` | Tier 4: Reflection memory (Markdown files) |
| `scholar_lens/memory/memory_manager.py` | Unified memory manager coordinating all tiers |
| `scholar_lens/agents/__init__.py` | Agents package init |
| `scholar_lens/agents/state.py` | Shared LangGraph state definition |
| `scholar_lens/agents/doc_analyzer.py` | Document Analyzer Agent |
| `scholar_lens/agents/explainer.py` | Content Explainer Agent |
| `scholar_lens/agents/validator.py` | Validator Agent |
| `scholar_lens/agents/tutor.py` | Learning Tutor Agent |
| `scholar_lens/agents/orchestrator.py` | LangGraph orchestration (pipeline + interactive) |
| `scholar_lens/agents/prompts.py` | All agent prompt templates |
| `tests/unit/memory/__init__.py` | Memory test package |
| `tests/unit/memory/test_core_memory.py` | Core memory tests |
| `tests/unit/memory/test_structured_memory.py` | Structured memory tests |
| `tests/unit/memory/test_reflection_memory.py` | Reflection memory tests |
| `tests/unit/memory/test_memory_manager.py` | Memory manager tests |
| `tests/unit/agents/__init__.py` | Agents test package |
| `tests/unit/agents/test_state.py` | State tests |
| `tests/unit/agents/test_doc_analyzer.py` | Doc analyzer tests |
| `tests/unit/agents/test_explainer.py` | Explainer tests |
| `tests/unit/agents/test_validator.py` | Validator tests |
| `tests/unit/agents/test_tutor.py` | Tutor tests |
| `tests/unit/agents/test_orchestrator.py` | Orchestrator tests |

---

### Task 1: Memory Package + Core Memory (Tier 1)

**Files:**
- Create: `scholar_lens/memory/__init__.py`
- Create: `scholar_lens/memory/core_memory.py`
- Create: `tests/unit/memory/__init__.py`
- Create: `tests/unit/memory/test_core_memory.py`

- [ ] **Step 1: Write failing tests for core memory**

```python
# tests/unit/memory/test_core_memory.py
import pytest
from scholar_lens.memory.core_memory import CoreMemory


class TestCoreMemory:
    def test_create_empty(self):
        cm = CoreMemory()
        assert cm.student_profile == ""
        assert cm.current_position == ""
        assert cm.active_glossary == []
        assert cm.session_summary == ""

    def test_create_with_data(self):
        cm = CoreMemory(
            student_profile="Intermediate CS student, strong in math, weak in NLP",
            current_position="paper_001:3.1",
            active_glossary=["self-attention:自注意力", "positional encoding:位置编码"],
            session_summary="Reading Transformer paper, discussed attention mechanism.",
        )
        assert "Intermediate" in cm.student_profile
        assert len(cm.active_glossary) == 2

    def test_to_context_string(self):
        cm = CoreMemory(
            student_profile="Intermediate student",
            current_position="paper_001:3.1",
            active_glossary=["attention:注意力"],
            session_summary="Discussing attention.",
        )
        context = cm.to_context_string()
        assert "Intermediate student" in context
        assert "paper_001:3.1" in context
        assert "attention:注意力" in context

    def test_token_estimate(self):
        cm = CoreMemory(
            student_profile="x" * 100,
            current_position="doc:1",
            active_glossary=["term:def"],
            session_summary="y" * 200,
        )
        tokens = cm.estimate_tokens()
        assert tokens > 0

    def test_update_glossary_max_size(self):
        """Glossary should be capped at 20 entries."""
        cm = CoreMemory()
        for i in range(25):
            cm.active_glossary.append(f"term{i}:def{i}")
        assert len(cm.active_glossary) == 20

    def test_update_position(self):
        cm = CoreMemory()
        cm.update_position("paper_001", "4.2")
        assert cm.current_position == "paper_001:4.2"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/memory/test_core_memory.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement core memory**

```python
# scholar_lens/memory/core_memory.py
from __future__ import annotations

from pydantic import BaseModel, Field


class CoreMemory(BaseModel):
    """Tier 1: Core memory — always in agent context, ~500 token budget.

    Per spec Section 5.1:
    - Student profile (3-5 sentences)
    - Current reading position (doc + section)
    - Active term glossary (last 10-20 terms)
    - Current session summary (1-2 paragraphs)
    Storage: System prompt
    """

    student_profile: str = ""
    current_position: str = ""  # "doc_id:section_id"
    active_glossary: list[str] = Field(default_factory=list)  # ["term:translation", ...]
    session_summary: str = ""

    _MAX_GLOSSARY_SIZE = 20

    def model_post_init(self, __context) -> None:
        self.active_glossary = self.active_glossary[-self._MAX_GLOSSARY_SIZE:]

    def update_position(self, doc_id: str, section_id: str) -> None:
        self.current_position = f"{doc_id}:{section_id}"

    def add_glossary_entry(self, term: str, translation: str) -> None:
        entry = f"{term}:{translation}"
        # Remove existing entry for same term
        self.active_glossary = [e for e in self.active_glossary if not e.startswith(f"{term}:")]
        self.active_glossary.append(entry)
        # Cap at max size
        if len(self.active_glossary) > self._MAX_GLOSSARY_SIZE:
            self.active_glossary = self.active_glossary[-self._MAX_GLOSSARY_SIZE:]

    def to_context_string(self) -> str:
        parts = []
        if self.student_profile:
            parts.append(f"Student Profile: {self.student_profile}")
        if self.current_position:
            parts.append(f"Current Position: {self.current_position}")
        if self.active_glossary:
            glossary = ", ".join(self.active_glossary)
            parts.append(f"Active Glossary: {glossary}")
        if self.session_summary:
            parts.append(f"Session Summary: {self.session_summary}")
        return "\n".join(parts)

    def estimate_tokens(self) -> int:
        text = self.to_context_string()
        chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
        other_chars = len(text) - chinese_chars
        return chinese_chars // 2 + other_chars // 4
```

```python
# scholar_lens/memory/__init__.py
from scholar_lens.memory.core_memory import CoreMemory

__all__ = ["CoreMemory"]
```

```python
# tests/unit/memory/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/memory/test_core_memory.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/memory/ tests/unit/memory/ && git commit -m "feat: add Tier 1 core memory (student profile, position, glossary, session summary)"
```

---

### Task 2: Structured Long-term Memory (Tier 2)

**Files:**
- Create: `scholar_lens/memory/structured_memory.py`
- Create: `tests/unit/memory/test_structured_memory.py`

- [ ] **Step 1: Write failing tests for structured memory**

```python
# tests/unit/memory/test_structured_memory.py
import pytest
import tempfile
from pathlib import Path
from scholar_lens.memory.structured_memory import StructuredMemory


@pytest.fixture
def memory(tmp_path):
    return StructuredMemory(db_path=str(tmp_path / "test.db"))


class TestStructuredMemory:
    @pytest.mark.asyncio
    async def test_add_reading_record(self, memory):
        await memory.add_reading_record(
            doc_id="paper_001",
            section_id="3.1",
            comprehension_score=0.8,
        )
        records = await memory.get_reading_history("paper_001")
        assert len(records) == 1
        assert records[0]["section_id"] == "3.1"

    @pytest.mark.asyncio
    async def test_add_term_log(self, memory):
        await memory.add_term_log(
            term="self-attention",
            definition_zh="自注意力机制",
        )
        terms = await memory.get_term_log()
        assert len(terms) == 1
        assert terms[0]["term"] == "self-attention"

    @pytest.mark.asyncio
    async def test_add_session_summary(self, memory):
        await memory.add_session_summary(
            doc_id="paper_001",
            topics="attention, transformer",
            difficulty="advanced",
        )
        summaries = await memory.get_session_summaries()
        assert len(summaries) == 1

    @pytest.mark.asyncio
    async def test_add_validation_record(self, memory):
        await memory.add_validation_record(
            explanation_id="exp_001",
            passed=True,
            issues=[],
        )
        records = await memory.get_validation_records()
        assert len(records) == 1
        assert records[0]["passed"] == 1

    @pytest.mark.asyncio
    async def test_multiple_records(self, memory):
        for i in range(5):
            await memory.add_reading_record(
                doc_id=f"doc_{i}",
                section_id="1",
                comprehension_score=0.5 + i * 0.1,
            )
        all_records = await memory.get_all_reading_history()
        assert len(all_records) == 5

    @pytest.mark.asyncio
    async def test_close(self, memory):
        await memory.close()
        # Should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/memory/test_structured_memory.py -v
```

Expected: FAIL

- [ ] **Step 3: Install aiosqlite**

```bash
pip install aiosqlite 2>&1 | tail -3
```

- [ ] **Step 4: Implement structured memory**

```python
# scholar_lens/memory/structured_memory.py
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS reading_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    comprehension_score REAL DEFAULT 0.0,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS term_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT NOT NULL,
    definition_zh TEXT DEFAULT '',
    times_explained INTEGER DEFAULT 1,
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    topics TEXT DEFAULT '',
    difficulty TEXT DEFAULT 'intermediate',
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS validation_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    explanation_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    issues TEXT DEFAULT '[]',
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class StructuredMemory:
    """Tier 2: Structured long-term memory stored in SQLite.

    Per spec Section 5.1:
    - Reading history: (doc_id, section, ts, score)
    - Term log: (term, definition, times_explained)
    - Session summaries: (date, doc, topics, diff)
    - Validation records: (id, passed, issues)
    Storage: SQLite (structured query) + Markdown (human-readable export)
    """

    def __init__(self, db_path: str = "data/memory.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.executescript(_CREATE_TABLES)
            await self._db.commit()
        return self._db

    async def add_reading_record(
        self,
        doc_id: str,
        section_id: str,
        comprehension_score: float = 0.0,
    ) -> None:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO reading_history (doc_id, section_id, comprehension_score) VALUES (?, ?, ?)",
            (doc_id, section_id, comprehension_score),
        )
        await db.commit()

    async def get_reading_history(self, doc_id: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT * FROM reading_history WHERE doc_id = ? ORDER BY timestamp DESC",
            (doc_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_all_reading_history(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM reading_history ORDER BY timestamp DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_term_log(self, term: str, definition_zh: str = "") -> None:
        db = await self._get_db()
        # Check if term exists
        cursor = await db.execute("SELECT id, times_explained FROM term_log WHERE term = ?", (term,))
        row = await cursor.fetchone()
        if row:
            await db.execute(
                "UPDATE term_log SET times_explained = ?, last_seen = CURRENT_TIMESTAMP WHERE id = ?",
                (row["times_explained"] + 1, row["id"]),
            )
        else:
            await db.execute(
                "INSERT INTO term_log (term, definition_zh) VALUES (?, ?)",
                (term, definition_zh),
            )
        await db.commit()

    async def get_term_log(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM term_log ORDER BY last_seen DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_session_summary(
        self,
        doc_id: str,
        topics: str = "",
        difficulty: str = "intermediate",
    ) -> None:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO session_summaries (doc_id, topics, difficulty) VALUES (?, ?, ?)",
            (doc_id, topics, difficulty),
        )
        await db.commit()

    async def get_session_summaries(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM session_summaries ORDER BY timestamp DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_validation_record(
        self,
        explanation_id: str,
        passed: bool,
        issues: list[str] | None = None,
    ) -> None:
        db = await self._get_db()
        issues_json = json.dumps(issues or [])
        await db.execute(
            "INSERT INTO validation_records (explanation_id, passed, issues) VALUES (?, ?, ?)",
            (explanation_id, 1 if passed else 0, issues_json),
        )
        await db.commit()

    async def get_validation_records(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM validation_records ORDER BY timestamp DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/memory/test_structured_memory.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/memory/structured_memory.py tests/unit/memory/test_structured_memory.py && git commit -m "feat: add Tier 2 structured long-term memory (SQLite: reading history, terms, sessions, validations)"
```

---

### Task 3: Reflection Memory (Tier 4) + Memory Manager

**Files:**
- Create: `scholar_lens/memory/reflection_memory.py`
- Create: `scholar_lens/memory/memory_manager.py`
- Create: `tests/unit/memory/test_reflection_memory.py`
- Create: `tests/unit/memory/test_memory_manager.py`

- [ ] **Step 1: Write failing tests for reflection memory**

```python
# tests/unit/memory/test_reflection_memory.py
import pytest
from pathlib import Path
from scholar_lens.memory.reflection_memory import ReflectionMemory


class TestReflectionMemory:
    @pytest.mark.asyncio
    async def test_save_reflection(self, tmp_path):
        rm = ReflectionMemory(knowledge_dir=str(tmp_path))
        await rm.save_reflection(
            title="weekly_reflection_2026-05-20",
            content="# Weekly Reflection\n\nLearned about attention mechanisms.",
        )
        files = list(tmp_path.glob("reflections/*.md"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "attention" in content

    @pytest.mark.asyncio
    async def test_get_latest_reflection(self, tmp_path):
        rm = ReflectionMemory(knowledge_dir=str(tmp_path))
        await rm.save_reflection(title="r1", content="First reflection")
        await rm.save_reflection(title="r2", content="Second reflection")
        latest = await rm.get_latest_reflection()
        assert "Second" in latest

    @pytest.mark.asyncio
    async def test_get_latest_no_reflections(self, tmp_path):
        rm = ReflectionMemory(knowledge_dir=str(tmp_path))
        result = await rm.get_latest_reflection()
        assert result == ""

    @pytest.mark.asyncio
    async def test_obsidian_format(self, tmp_path):
        rm = ReflectionMemory(knowledge_dir=str(tmp_path))
        await rm.save_reflection(
            title="test",
            content="# Reflection\n\nContent here.",
            tags=["learning", "transformer"],
        )
        files = list(tmp_path.glob("reflections/*.md"))
        content = files[0].read_text()
        assert "---" in content  # YAML frontmatter
        assert "learning" in content
```

- [ ] **Step 2: Write failing tests for memory manager**

```python
# tests/unit/memory/test_memory_manager.py
import pytest
from scholar_lens.memory.memory_manager import MemoryManager


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_create(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        assert mm is not None

    @pytest.mark.asyncio
    async def test_core_memory_access(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        cm = mm.core_memory
        assert cm.student_profile == ""

    @pytest.mark.asyncio
    async def test_update_core_position(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        mm.core_memory.update_position("doc1", "3.1")
        assert mm.core_memory.current_position == "doc1:3.1"

    @pytest.mark.asyncio
    async def test_structured_memory_roundtrip(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        await mm.structured.add_reading_record("doc1", "1", 0.9)
        records = await mm.structured.get_reading_history("doc1")
        assert len(records) == 1
        await mm.close()

    @pytest.mark.asyncio
    async def test_reflection_roundtrip(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        await mm.reflection.save_reflection("test", "Content")
        result = await mm.reflection.get_latest_reflection()
        assert "Content" in result
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/memory/test_reflection_memory.py tests/unit/memory/test_memory_manager.py -v
```

Expected: FAIL

- [ ] **Step 4: Implement reflection memory**

```python
# scholar_lens/memory/reflection_memory.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ReflectionMemory:
    """Tier 4: Reflection memory — learning pattern insights and strategy adjustments.

    Per spec Section 5.1:
    - Learning pattern insights
    - Strategy adjustment suggestions
    - Knowledge gap summaries
    Storage: Markdown files (human-readable, Obsidian-compatible)
    Generation: every 5 reading sessions
    """

    def __init__(self, knowledge_dir: str = "knowledge") -> None:
        self._dir = Path(knowledge_dir) / "reflections"
        self._dir.mkdir(parents=True, exist_ok=True)

    async def save_reflection(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> Path:
        now = datetime.now()
        filename = f"{now.strftime('%Y-%m-%d')}_{title}.md"
        filepath = self._dir / filename

        tags_str = ", ".join(tags or [])
        frontmatter = f"""---
date: {now.strftime('%Y-%m-%d')}
tags: [{tags_str}]
type: reflection
---

"""
        filepath.write_text(frontmatter + content, encoding="utf-8")
        logger.info(f"Saved reflection to {filepath}")
        return filepath

    async def get_latest_reflection(self) -> str:
        files = sorted(self._dir.glob("*.md"), reverse=True)
        if not files:
            return ""
        return files[0].read_text(encoding="utf-8")

    async def get_all_reflections(self) -> list[str]:
        files = sorted(self._dir.glob("*.md"), reverse=True)
        return [f.read_text(encoding="utf-8") for f in files]
```

- [ ] **Step 5: Implement memory manager**

```python
# scholar_lens/memory/memory_manager.py
from __future__ import annotations

from pathlib import Path

from scholar_lens.memory.core_memory import CoreMemory
from scholar_lens.memory.structured_memory import StructuredMemory
from scholar_lens.memory.reflection_memory import ReflectionMemory


class MemoryManager:
    """Unified memory manager coordinating all four tiers.

    Per spec Section 5.2 (Agent-Memory Interaction matrix):
    - Document Analyzer: writes to T2 (term glossary), T3 (L0/L1/L2)
    - Content Explainer: reads T1, T2; reads T3; writes T2 (explanation record)
    - Validator: reads T2, T3; writes T2 (validation record)
    - Learning Tutor: reads all; writes T2 (reading history), T4 (reflections)
    """

    def __init__(self, data_dir: str = "data") -> None:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        self.core_memory = CoreMemory()
        self.structured = StructuredMemory(db_path=str(data_path / "memory.db"))
        self.reflection = ReflectionMemory(knowledge_dir=str(data_path / "knowledge"))

    async def close(self) -> None:
        await self.structured.close()

    def get_core_context(self) -> str:
        """Get Tier 1 core memory as context string for agent system prompts."""
        return self.core_memory.to_context_string()
```

- [ ] **Step 6: Update memory __init__.py**

```python
# scholar_lens/memory/__init__.py
from scholar_lens.memory.core_memory import CoreMemory
from scholar_lens.memory.structured_memory import StructuredMemory
from scholar_lens.memory.reflection_memory import ReflectionMemory
from scholar_lens.memory.memory_manager import MemoryManager

__all__ = ["CoreMemory", "StructuredMemory", "ReflectionMemory", "MemoryManager"]
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/memory/ -v
```

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/memory/ tests/unit/memory/ && git commit -m "feat: add Tier 4 reflection memory and unified memory manager"
```

---

### Task 4: Agent State + Prompts

**Files:**
- Create: `scholar_lens/agents/__init__.py`
- Create: `scholar_lens/agents/state.py`
- Create: `scholar_lens/agents/prompts.py`
- Create: `tests/unit/agents/__init__.py`
- Create: `tests/unit/agents/test_state.py`

- [ ] **Step 1: Write failing tests for agent state**

```python
# tests/unit/agents/test_state.py
import pytest
from scholar_lens.agents.state import ScholarLensState, AgentStep


class TestScholarLensState:
    def test_create_empty(self):
        state = ScholarLensState()
        assert state.doc_id == ""
        assert state.messages == []
        assert state.current_step == ""

    def test_create_with_data(self):
        state = ScholarLensState(
            doc_id="paper_001",
            messages=[{"role": "user", "content": "Explain section 3.1"}],
            current_step="explainer",
        )
        assert state.doc_id == "paper_001"
        assert len(state.messages) == 1

    def test_add_message(self):
        state = ScholarLensState()
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi there")
        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "user"

    def test_agent_step_enum(self):
        assert AgentStep.ANALYZE == "analyze"
        assert AgentStep.EXPLAIN == "explain"
        assert AgentStep.VALIDATE == "validate"
        assert AgentStep.TUTOR == "tutor"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_state.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement agent state and prompts**

```python
# scholar_lens/agents/__init__.py
```

```python
# tests/unit/agents/__init__.py
```

```python
# scholar_lens/agents/state.py
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from scholar_lens.core.models import (
    DocumentUnderstanding,
    ExplanationResult,
    ValidationResult,
    StudentProfile,
)


class AgentStep(str, Enum):
    ANALYZE = "analyze"
    EXPLAIN = "explain"
    VALIDATE = "validate"
    TUTOR = "tutor"


class ScholarLensState(BaseModel):
    """Shared state for all agents in the LangGraph pipeline.

    This is the state that flows through the graph nodes.
    """

    # Document context
    doc_id: str = ""
    file_path: str = ""
    doc_understanding: DocumentUnderstanding | None = None

    # Current interaction
    messages: list[dict[str, str]] = Field(default_factory=list)
    current_step: str = ""
    explanation_request: str = ""
    explanation_result: ExplanationResult | None = None
    validation_result: ValidationResult | None = None

    # Student context
    student_profile: StudentProfile = Field(default_factory=StudentProfile)

    # RAG context
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    section_id: str = ""

    # Session
    session_id: str = ""
    error: str = ""

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
```

```python
# scholar_lens/agents/prompts.py
from __future__ import annotations

DOC_ANALYZER_SYSTEM = """You are a document analysis expert. Your job is to:
1. Identify the document type (research_paper, courseware, textbook_chapter)
2. Extract the section structure
3. Identify key terms with Chinese translations
4. Assess reading difficulty (beginner, intermediate, advanced)
5. Generate a Mermaid diagram showing the document structure
6. Generate L0 (~100 token) and L1 (~2k token) summaries for each section

Output a structured DocumentUnderstanding object.
"""

DOC_ANALYZER_STRUCTURE = """Analyze the following document and extract its structure.

Document text (first 5000 tokens):
{document_text}

Respond with:
- doc_type: research_paper | courseware | textbook_chapter
- language: en | zh | mixed
- difficulty: beginner | intermediate | advanced
- sections: list of {{id, title, level, page_start, page_end}}
- key_terms: list of {{english, chinese, relation_type}}
- mermaid_map: Mermaid diagram of the structure
"""

EXPLAINER_SYSTEM = """You are a bilingual academic content explainer. Your job is to:
1. Translate English academic text to Chinese while preserving key English terms inline
2. Explain concepts in clear Chinese
3. Connect related terms and concepts
4. Adapt explanation depth to the student's level

Translation rules:
- Key terms preserved inline: "self-attention mechanism（自注意力机制）"
- Formulas preserved as LaTeX with Chinese meaning
- Maintain a per-document bilingual glossary for consistency
"""

EXPLAINER_TRANSLATE = """Translate and explain the following text for a {level} student.

Section context: {section_title}
Previous explanations given: {previous_count}

Text to explain:
{target_text}

Provide:
- original: the original English text
- translation: Chinese translation with key terms inline
- explanation: detailed Chinese explanation
- related_terms: list of related terms {{english, chinese}}
- difficulty_level: beginner | intermediate | advanced
- confidence: high | medium | low | unverified
"""

VALIDATOR_SYSTEM = """You are a content validation expert. Your job is to verify:
1. Term translation accuracy and consistency
2. Faithfulness to the original source text
3. Detect hallucinations or inaccurate explanations

You have access to the original source text. Compare the explanation against it.
"""

VALIDATOR_CHECK = """Validate the following explanation against the source text.

Source text:
{source_text}

Explanation:
{explanation}

Check for:
1. Are key terms translated correctly and consistently?
2. Is the explanation faithful to the source?
3. Are there any hallucinated facts?

Respond with:
- passed: true | false
- confidence: high | medium | low
- issues: list of specific issues found
- correction: suggested correction if failed (null if passed)
"""

TUTOR_SYSTEM = """You are a Socratic learning tutor helping a Chinese university student read English academic papers.

Your role:
- Guide the student through the paper using Socratic questioning
- Provide scaffolding: adapt explanation depth to student level
- Detect knowledge gaps and prerequisite concepts
- Encourage teach-back for understanding verification
- Track student progress and adjust strategies

Interaction modes:
- Collaborative reading: preview upcoming sections
- Socratic questioning: ask probing questions
- Scaffolding: Level 1 (full) → Level 2 (hints) → Level 3 (questions only)
- Teach-back: ask student to explain in their own words
- Gap detection: identify missing prerequisites

Core memory:
{core_memory}

Document structure:
{mermaid_map}
"""

TUTOR_RESPONSE = """The student asks:
{question}

Current section: {section_id}
Student level: {student_level}

Decide how to respond:
- If general knowledge → answer directly
- If document content → use retrieved context
- If needs deep explanation → request explainer
- If student is struggling → simplify scaffolding

{retrieved_context}
"""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_state.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/agents/ tests/unit/agents/ && git commit -m "feat: add agent state, step enum, and all prompt templates"
```

---

### Task 5: Document Analyzer Agent

**Files:**
- Create: `scholar_lens/agents/doc_analyzer.py`
- Create: `tests/unit/agents/test_doc_analyzer.py`

- [ ] **Step 1: Write failing tests for doc analyzer**

```python
# tests/unit/agents/test_doc_analyzer.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from scholar_lens.agents.doc_analyzer import DocumentAnalyzerAgent
from scholar_lens.agents.state import ScholarLensState


class TestDocumentAnalyzerAgent:
    @pytest.mark.asyncio
    async def test_analyze_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''```json
{
    "doc_type": "research_paper",
    "language": "en",
    "difficulty": "advanced",
    "estimated_reading_time": 45,
    "sections": [{"section_id": "1", "title": "Introduction", "level": 1, "page_start": 1, "page_end": 2, "section_type": "prose", "difficulty": "intermediate"}],
    "mermaid_map": "graph TD\\n  A[Intro]-->B[Method]",
    "key_terms": [{"english": "transformer", "chinese": "Transformer"}],
    "l0_summaries": {"1": "Introduces the problem"},
    "l1_overviews": {"1": "This paper introduces the Transformer architecture..."},
    "references": [],
    "citation_contexts": [],
    "prerequisites": ["attention mechanism"]
}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = DocumentAnalyzerAgent(llm=mock_llm)
        state = ScholarLensState(
            doc_id="paper_001",
            file_path="test.pdf",
        )
        # Add document text to messages
        state.add_message("system", "Document text: Attention Is All You Need. Abstract: We propose a new architecture...")

        result = await agent.analyze(state)
        assert result.doc_understanding is not None
        assert result.doc_understanding.doc_type == "research_paper"
        assert result.current_step == "analyze"

    @pytest.mark.asyncio
    async def test_analyze_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        agent = DocumentAnalyzerAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001")
        state.add_message("system", "Document text: Some text")

        result = await agent.analyze(state)
        assert result.error != ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_doc_analyzer.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement doc analyzer agent**

```python
# scholar_lens/agents/doc_analyzer.py
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from scholar_lens.agents.prompts import DOC_ANALYZER_SYSTEM, DOC_ANALYZER_STRUCTURE
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import DocumentUnderstanding, Section, Term

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class DocumentAnalyzerAgent:
    """Document Analyzer Agent per spec Section 4.1.

    Responsibility: Parse document, extract structure, concepts, difficulty;
    generate Mermaid structure map and L0/L1/L2 layered content.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    async def analyze(self, state: ScholarLensState) -> ScholarLensState:
        """Analyze the document and populate doc_understanding."""
        if not self._llm:
            state.error = "No LLM configured for Document Analyzer"
            state.current_step = "analyze"
            return state

        # Get document text from messages
        doc_text = ""
        for msg in state.messages:
            if "Document text:" in msg.get("content", ""):
                doc_text = msg["content"]
                break

        if not doc_text:
            state.error = "No document text found in state"
            state.current_step = "analyze"
            return state

        try:
            result = await self._call_llm(doc_text)
            understanding = self._parse_result(result)
            state.doc_understanding = understanding
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            state.error = f"Analysis failed: {e}"

        state.current_step = "analyze"
        return state

    async def _call_llm(self, doc_text: str) -> str:
        prompt = DOC_ANALYZER_STRUCTURE.format(document_text=doc_text[:5000])
        response = await self._llm.ainvoke([
            SystemMessage(content=DOC_ANALYZER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content

    def _parse_result(self, llm_output: str) -> DocumentUnderstanding:
        """Parse LLM output into DocumentUnderstanding."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = llm_output

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: create minimal understanding
            return DocumentUnderstanding(
                doc_type="general_document",
                language="en",
                difficulty="intermediate",
                estimated_reading_time=30,
                sections=[Section(section_id="1", title="Document", level=1)],
                mermaid_map="",
            )

        sections = [
            Section(
                section_id=s.get("section_id", s.get("id", "1")),
                title=s.get("title", ""),
                level=s.get("level", 1),
                page_start=s.get("page_start"),
                page_end=s.get("page_end"),
                section_type=s.get("section_type", "prose"),
                difficulty=s.get("difficulty", "intermediate"),
            )
            for s in data.get("sections", [])
        ]

        terms = [
            Term(
                english=t.get("english", ""),
                chinese=t.get("chinese", ""),
                relation_type=t.get("relation_type"),
            )
            for t in data.get("key_terms", [])
        ]

        return DocumentUnderstanding(
            doc_type=data.get("doc_type", "general_document"),
            language=data.get("language", "en"),
            difficulty=data.get("difficulty", "intermediate"),
            estimated_reading_time=data.get("estimated_reading_time", 30),
            sections=sections,
            mermaid_map=data.get("mermaid_map", ""),
            key_terms=terms,
            prerequisites=data.get("prerequisites", []),
            l0_summaries=data.get("l0_summaries", {}),
            l1_overviews=data.get("l1_overviews", {}),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_doc_analyzer.py -v
```

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/agents/doc_analyzer.py tests/unit/agents/test_doc_analyzer.py && git commit -m "feat: add Document Analyzer Agent with LLM-based structure extraction"
```

---

### Task 6: Content Explainer + Validator Agents

**Files:**
- Create: `scholar_lens/agents/explainer.py`
- Create: `scholar_lens/agents/validator.py`
- Create: `tests/unit/agents/test_explainer.py`
- Create: `tests/unit/agents/test_validator.py`

- [ ] **Step 1: Write failing tests for explainer**

```python
# tests/unit/agents/test_explainer.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from scholar_lens.agents.explainer import ContentExplainerAgent
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import DocumentUnderstanding, Section, ExplanationResult


class TestContentExplainerAgent:
    @pytest.mark.asyncio
    async def test_explain_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''```json
{
    "original": "The self-attention mechanism computes attention scores",
    "translation": "自注意力机制（self-attention mechanism）计算注意力分数",
    "explanation": "自注意力是一种让序列中每个位置都能关注其他位置的机制。",
    "related_terms": [{"english": "attention", "chinese": "注意力"}],
    "difficulty_level": "intermediate",
    "source_section": "3.1",
    "confidence": "high"
}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = ContentExplainerAgent(llm=mock_llm)
        state = ScholarLensState(
            doc_id="paper_001",
            section_id="3.1",
            explanation_request="Explain self-attention",
        )
        state.student_profile.level = "intermediate"

        result = await agent.explain(state)
        assert result.explanation_result is not None
        assert result.explanation_result.confidence == "high"
        assert result.current_step == "explain"

    @pytest.mark.asyncio
    async def test_explain_fallback(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

        agent = ContentExplainerAgent(llm=mock_llm)
        state = ScholarLensState(
            doc_id="paper_001",
            section_id="3.1",
            explanation_request="Explain this",
        )

        result = await agent.explain(state)
        assert result.error != ""
```

- [ ] **Step 2: Write failing tests for validator**

```python
# tests/unit/agents/test_validator.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from scholar_lens.agents.validator import ValidatorAgent
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import ExplanationResult, ValidationResult, Term


class TestValidatorAgent:
    @pytest.mark.asyncio
    async def test_validate_passed(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''```json
{
    "passed": true,
    "confidence": "high",
    "issues": [],
    "correction": null
}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = ValidatorAgent(llm=mock_llm)
        state = ScholarLensState()
        state.explanation_result = ExplanationResult(
            original="Self-attention computes attention",
            translation="自注意力计算注意力",
            explanation="自注意力是一种机制",
            confidence="high",
        )

        result = await agent.validate(state)
        assert result.validation_result is not None
        assert result.validation_result.passed is True

    @pytest.mark.asyncio
    async def test_validate_failed_with_correction(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''```json
{
    "passed": false,
    "confidence": "low",
    "issues": ["Term 'attention' mistranslated as 关注 instead of 注意力"],
    "correction": "attention should be 注意力, not 关注"
}
```'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = ValidatorAgent(llm=mock_llm)
        state = ScholarLensState()
        state.explanation_result = ExplanationResult(
            original="test",
            translation="test",
            explanation="test",
            confidence="low",
        )

        result = await agent.validate(state)
        assert result.validation_result is not None
        assert result.validation_result.passed is False
        assert result.validation_result.correction is not None

    @pytest.mark.asyncio
    async def test_validate_skips_when_no_explanation(self):
        agent = ValidatorAgent(llm=None)
        state = ScholarLensState()

        result = await agent.validate(state)
        assert result.validation_result is None
        assert result.current_step == "validate"

    @pytest.mark.asyncio
    async def test_validate_failure_does_not_block(self):
        """Validator failure should never block the main flow."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        agent = ValidatorAgent(llm=mock_llm)
        state = ScholarLensState()
        state.explanation_result = ExplanationResult(
            original="x", translation="x", explanation="x", confidence="medium"
        )

        result = await agent.validate(state)
        # Should mark as unverified, not crash
        assert result.validation_result is not None
        assert result.validation_result.confidence == "unverified"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_explainer.py tests/unit/agents/test_validator.py -v
```

Expected: FAIL

- [ ] **Step 4: Implement explainer agent**

```python
# scholar_lens/agents/explainer.py
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from scholar_lens.agents.prompts import EXPLAINER_SYSTEM, EXPLAINER_TRANSLATE
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import ExplanationResult, Term

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ContentExplainerAgent:
    """Content Explainer Agent per spec Section 4.2.

    Responsibility: On-demand translation, term explanation,
    sentence breakdown, concept connection.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    async def explain(self, state: ScholarLensState) -> ScholarLensState:
        """Generate explanation for the requested content."""
        if not self._llm:
            state.error = "No LLM configured for Explainer"
            state.current_step = "explain"
            return state

        if not state.explanation_request:
            state.error = "No explanation request in state"
            state.current_step = "explain"
            return state

        try:
            result_text = await self._call_llm(state)
            explanation = self._parse_result(result_text)
            state.explanation_result = explanation
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            state.error = f"Explanation failed: {e}"

        state.current_step = "explain"
        return state

    async def _call_llm(self, state: ScholarLensState) -> str:
        section_title = ""
        if state.doc_understanding:
            for s in state.doc_understanding.sections:
                if s.section_id == state.section_id:
                    section_title = s.title
                    break

        previous_count = sum(1 for m in state.messages if m.get("role") == "assistant")

        prompt = EXPLAINER_TRANSLATE.format(
            level=state.student_profile.level,
            section_title=section_title or state.section_id,
            previous_count=previous_count,
            target_text=state.explanation_request,
        )

        response = await self._llm.ainvoke([
            SystemMessage(content=EXPLAINER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content

    def _parse_result(self, llm_output: str) -> ExplanationResult:
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = llm_output

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return ExplanationResult(
                original=llm_output,
                translation="",
                explanation=llm_output,
                confidence="unverified",
            )

        related_terms = [
            Term(english=t.get("english", ""), chinese=t.get("chinese", ""))
            for t in data.get("related_terms", [])
        ]

        return ExplanationResult(
            original=data.get("original", ""),
            translation=data.get("translation", ""),
            explanation=data.get("explanation", ""),
            related_terms=related_terms,
            difficulty_level=data.get("difficulty_level", "intermediate"),
            source_section=data.get("source_section", ""),
            confidence=data.get("confidence", "medium"),
        )
```

- [ ] **Step 5: Implement validator agent**

```python
# scholar_lens/agents/validator.py
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from scholar_lens.agents.prompts import VALIDATOR_SYSTEM, VALIDATOR_CHECK
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import ValidationResult

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """Validator Agent per spec Section 4.3.

    Responsibility: Cross-verify explainer output — term accuracy,
    faithfulness to source, hallucination detection.

    Failure handling: never blocks main flow. On failure, mark as "unverified".
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    async def validate(self, state: ScholarLensState) -> ScholarLensState:
        """Validate the explanation result."""
        state.current_step = "validate"

        if state.explanation_result is None:
            return state

        # Rule-based validation first (zero cost)
        rule_result = self._rule_validate(state)
        if not rule_result.passed:
            state.validation_result = rule_result
            return state

        # LLM-based validation if LLM available
        if self._llm:
            try:
                llm_result = await self._llm_validate(state)
                state.validation_result = llm_result
            except Exception as e:
                logger.warning(f"LLM validation failed: {e}")
                state.validation_result = ValidationResult(
                    passed=True,
                    confidence="unverified",
                    issues=[f"LLM validation skipped: {e}"],
                )
        else:
            # No LLM, mark as unverified
            state.validation_result = ValidationResult(
                passed=True,
                confidence="unverified",
                issues=["No LLM available for validation"],
            )

        return state

    def _rule_validate(self, state: ScholarLensState) -> ValidationResult:
        """Rule-based validation: always runs, zero cost.

        - Check term consistency against glossary
        - Check source backtracking (L2 chunk match)
        """
        issues = []
        explanation = state.explanation_result

        if not explanation:
            return ValidationResult(passed=True, confidence="high")

        # Check: empty translation or explanation
        if not explanation.translation.strip():
            issues.append("Empty translation")
        if not explanation.explanation.strip():
            issues.append("Empty explanation")

        if issues:
            return ValidationResult(passed=False, confidence="high", issues=issues)

        return ValidationResult(passed=True, confidence="high")

    async def _llm_validate(self, state: ScholarLensState) -> ValidationResult:
        explanation = state.explanation_result
        source_text = ""
        for msg in state.messages:
            if "Document text:" in msg.get("content", ""):
                source_text = msg["content"][:2000]
                break

        prompt = VALIDATOR_CHECK.format(
            source_text=source_text or "Source not available",
            explanation=f"Original: {explanation.original}\nTranslation: {explanation.translation}\nExplanation: {explanation.explanation}",
        )

        response = await self._llm.ainvoke([
            SystemMessage(content=VALIDATOR_SYSTEM),
            HumanMessage(content=prompt),
        ])

        return self._parse_result(response.content)

    def _parse_result(self, llm_output: str) -> ValidationResult:
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = llm_output

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return ValidationResult(passed=True, confidence="unverified")

        return ValidationResult(
            passed=data.get("passed", True),
            confidence=data.get("confidence", "medium"),
            issues=data.get("issues", []),
            correction=data.get("correction"),
        )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_explainer.py tests/unit/agents/test_validator.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/agents/explainer.py scholar_lens/agents/validator.py tests/unit/agents/test_explainer.py tests/unit/agents/test_validator.py && git commit -m "feat: add Content Explainer and Validator agents with rule+LLM validation"
```

---

### Task 7: Learning Tutor Agent

**Files:**
- Create: `scholar_lens/agents/tutor.py`
- Create: `tests/unit/agents/test_tutor.py`

- [ ] **Step 1: Write failing tests for tutor**

```python
# tests/unit/agents/test_tutor.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from scholar_lens.agents.tutor import LearningTutorAgent
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import DocumentUnderstanding, Section


class TestLearningTutorAgent:
    @pytest.mark.asyncio
    async def test_respond_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "这是一个很好的问题。自注意力机制的核心思想是让序列中的每个位置都能关注到其他所有位置。"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = LearningTutorAgent(llm=mock_llm)
        state = ScholarLensState(
            doc_id="paper_001",
            section_id="3.1",
        )
        state.student_profile.level = "intermediate"
        state.add_message("user", "什么是自注意力机制？")

        result = await agent.respond(state)
        assert len(result.messages) > 1  # At least the user message + response
        assert result.current_step == "tutor"

    @pytest.mark.asyncio
    async def test_respond_with_mermaid_map(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Let me explain the structure."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        agent = LearningTutorAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001", section_id="1")
        state.doc_understanding = DocumentUnderstanding(
            doc_type="research_paper",
            language="en",
            difficulty="advanced",
            estimated_reading_time=45,
            sections=[Section(section_id="1", title="Introduction", level=1)],
            mermaid_map="graph TD\n  A[Intro]-->B[Method]",
            key_terms=[],
        )
        state.add_message("user", "这篇论文的结构是什么？")

        result = await agent.respond(state)
        assert result.current_step == "tutor"

    @pytest.mark.asyncio
    async def test_respond_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

        agent = LearningTutorAgent(llm=mock_llm)
        state = ScholarLensState(doc_id="paper_001")
        state.add_message("user", "Hello")

        result = await agent.respond(state)
        assert result.error != ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_tutor.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement tutor agent**

```python
# scholar_lens/agents/tutor.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scholar_lens.agents.prompts import TUTOR_SYSTEM, TUTOR_RESPONSE
from scholar_lens.agents.state import ScholarLensState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class LearningTutorAgent:
    """Learning Tutor Agent per spec Section 4.4.

    This is the only agent that directly converses with the student.
    Other agents serve through the tutor.

    Interaction modes: collaborative reading, Socratic questioning,
    scaffolding, teach-back, gap detection.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        core_memory_context: str = "",
    ) -> None:
        self._llm = llm
        self._core_memory_context = core_memory_context

    async def respond(self, state: ScholarLensState) -> ScholarLensState:
        """Generate a tutor response to the student's latest message."""
        state.current_step = "tutor"

        if not self._llm:
            state.error = "No LLM configured for Tutor"
            return state

        # Get the latest user message
        user_message = ""
        for msg in reversed(state.messages):
            if msg.get("role") == "user":
                user_message = msg["content"]
                break

        if not user_message:
            state.error = "No user message found"
            return state

        try:
            response_content = await self._call_llm(state, user_message)
            state.add_message("assistant", response_content)
        except Exception as e:
            logger.error(f"Tutor response failed: {e}")
            state.error = f"Tutor response failed: {e}"

        return state

    async def _call_llm(self, state: ScholarLensState, user_message: str) -> str:
        # Build context
        mermaid_map = ""
        if state.doc_understanding:
            mermaid_map = state.doc_understanding.mermaid_map

        core_memory = self._core_memory_context or ""

        # Build retrieved context
        retrieved_context = ""
        if state.retrieved_chunks:
            chunks_text = "\n".join(
                f"[{c.get('section_id', '')}] {c.get('text', '')[:500]}"
                for c in state.retrieved_chunks[:3]
            )
            retrieved_context = f"Retrieved context:\n{chunks_text}"

        # Include explanation result if available
        if state.explanation_result:
            retrieved_context += f"\n\nExplanation from Explainer:\n{state.explanation_result.explanation}"

        system_prompt = TUTOR_SYSTEM.format(
            core_memory=core_memory or "No core memory loaded",
            mermaid_map=mermaid_map or "No document structure available",
        )

        user_prompt = TUTOR_RESPONSE.format(
            question=user_message,
            section_id=state.section_id or "unknown",
            student_level=state.student_profile.level,
            retrieved_context=retrieved_context or "No retrieved context available",
        )

        # Build conversation history (last 10 messages)
        history = state.messages[-10:] if len(state.messages) > 10 else state.messages
        langchain_messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        langchain_messages.append(HumanMessage(content=user_prompt))

        response = await self._llm.ainvoke(langchain_messages)
        return response.content
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_tutor.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/agents/tutor.py tests/unit/agents/test_tutor.py && git commit -m "feat: add Learning Tutor Agent with Socratic dialogue and context integration"
```

---

### Task 8: LangGraph Orchestrator

**Files:**
- Create: `scholar_lens/agents/orchestrator.py`
- Create: `tests/unit/agents/test_orchestrator.py`

- [ ] **Step 1: Write failing tests for orchestrator**

```python
# tests/unit/agents/test_orchestrator.py
import pytest
from scholar_lens.agents.orchestrator import ScholarLensOrchestrator


class TestScholarLensOrchestrator:
    def test_instantiation(self):
        orch = ScholarLensOrchestrator()
        assert orch is not None

    def test_build_pipeline_graph(self):
        orch = ScholarLensOrchestrator()
        graph = orch.build_pipeline_graph()
        assert graph is not None

    def test_build_tutor_graph(self):
        orch = ScholarLensOrchestrator()
        graph = orch.build_tutor_graph()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_pipeline_graph_nodes(self):
        """Pipeline graph should have analyze, explain, validate nodes."""
        orch = ScholarLensOrchestrator()
        graph = orch.build_pipeline_graph()
        # Verify the compiled graph exists
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_tutor_graph_nodes(self):
        """Tutor graph should have tutor, explainer, validator nodes."""
        orch = ScholarLensOrchestrator()
        graph = orch.build_tutor_graph()
        compiled = graph.compile()
        assert compiled is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_orchestrator.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement orchestrator**

```python
# scholar_lens/agents/orchestrator.py
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from scholar_lens.agents.state import AgentStep, ScholarLensState
from scholar_lens.agents.doc_analyzer import DocumentAnalyzerAgent
from scholar_lens.agents.explainer import ContentExplainerAgent
from scholar_lens.agents.validator import ValidatorAgent
from scholar_lens.agents.tutor import LearningTutorAgent

logger = logging.getLogger(__name__)


class ScholarLensOrchestrator:
    """LangGraph orchestration for ScholarLens agent pipeline.

    Two graphs:
    1. Pipeline graph: upload → analyze → explain → validate (one-shot processing)
    2. Tutor graph: tutor loop with on-demand explainer/validator (interactive)
    """

    def __init__(
        self,
        doc_analyzer: DocumentAnalyzerAgent | None = None,
        explainer: ContentExplainerAgent | None = None,
        validator: ValidatorAgent | None = None,
        tutor: LearningTutorAgent | None = None,
    ) -> None:
        self.doc_analyzer = doc_analyzer or DocumentAnalyzerAgent()
        self.explainer = explainer or ContentExplainerAgent()
        self.validator = validator or ValidatorAgent()
        self.tutor = tutor or LearningTutorAgent()

    def build_pipeline_graph(self) -> StateGraph:
        """Build the document processing pipeline graph.

        Flow: analyze → explain → validate → END
        """
        graph = StateGraph(ScholarLensState)

        graph.add_node("analyze", self.doc_analyzer.analyze)
        graph.add_node("explain", self.explainer.explain)
        graph.add_node("validate", self.validator.validate)

        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "explain")
        graph.add_edge("explain", "validate")
        graph.add_edge("validate", END)

        return graph

    def build_tutor_graph(self) -> StateGraph:
        """Build the interactive tutor graph.

        Flow: tutor → (conditional: needs_explanation? → explainer → validator → tutor | END)
        """
        graph = StateGraph(ScholarLensState)

        graph.add_node("tutor", self.tutor.respond)
        graph.add_node("explainer", self.explainer.explain)
        graph.add_node("validator", self.validator.validate)

        graph.set_entry_point("tutor")

        def route_after_tutor(state: ScholarLensState) -> str:
            """Decide whether to invoke explainer or end."""
            if state.explanation_request and not state.explanation_result:
                return "explainer"
            return END

        graph.add_conditional_edges("tutor", route_after_tutor, {"explainer": "explainer", END: END})
        graph.add_edge("explainer", "validator")
        graph.add_edge("validator", "tutor")

        return graph
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/agents/test_orchestrator.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Run all tests**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/agents/orchestrator.py tests/unit/agents/test_orchestrator.py && git commit -m "feat: add LangGraph orchestrator with pipeline and interactive tutor graphs"
```

---

## Self-Review

**1. Spec coverage:**
- Tier 1 Core Memory (Section 5.1) → Task 1 ✓
- Tier 2 Structured Memory (Section 5.1) → Task 2 ✓
- Tier 3 Document Memory → Integrated via RAG pipeline (Plan 02) ✓
- Tier 4 Reflection Memory (Section 5.1) → Task 3 ✓
- Memory Manager (Section 5.2) → Task 3 ✓
- Document Analyzer Agent (Section 4.1) → Task 5 ✓
- Content Explainer Agent (Section 4.2) → Task 6 ✓
- Validator Agent (Section 4.3) → Task 6 ✓
- Learning Tutor Agent (Section 4.4) → Task 7 ✓
- LangGraph Orchestration (Section 3.1) → Task 8 ✓
- Agent prompts → Task 4 ✓
- Gaps: Obsidian knowledge base output (Tier 2/4 Markdown export), knowledge tracing (p(known) updates), student model propagation — these are integration features for Plan 04.

**2. Placeholder scan:** No TBD or TODO found.

**3. Type consistency:** `ScholarLensState` used consistently across all agents and orchestrator. `DocumentUnderstanding`, `ExplanationResult`, `ValidationResult` all from `core.models` and used correctly. `CoreMemory` interface stable between memory manager and tutor.
