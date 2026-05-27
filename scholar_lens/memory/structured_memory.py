from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

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

CREATE TABLE IF NOT EXISTS learning_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    doc_id TEXT DEFAULT '',
    section_id TEXT DEFAULT '',
    payload TEXT DEFAULT '{}',
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS concept_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept TEXT NOT NULL,
    doc_id TEXT DEFAULT '',
    status TEXT DEFAULT 'learning',
    evidence_count INTEGER DEFAULT 1,
    source_events TEXT DEFAULT '[]',
    last_section_id TEXT DEFAULT '',
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(concept, doc_id)
);
"""


class StructuredMemory:
    """Tier 2: Structured long-term memory stored in SQLite."""

    def __init__(self, db_path: str = "data/memory.db") -> None:
        self._db_path = db_path
        self._db: sqlite3.Connection | None = None

    async def _get_db(self) -> sqlite3.Connection:
        if self._db is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self._db_path)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(_CREATE_TABLES)
            self._db.commit()
        return self._db

    async def add_reading_record(self, doc_id: str, section_id: str, comprehension_score: float = 0.0) -> None:
        db = await self._get_db()
        db.execute("INSERT INTO reading_history (doc_id, section_id, comprehension_score) VALUES (?, ?, ?)", (doc_id, section_id, comprehension_score))
        db.commit()

    async def get_reading_history(self, doc_id: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = db.execute("SELECT * FROM reading_history WHERE doc_id = ? ORDER BY timestamp DESC", (doc_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_all_reading_history(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = db.execute("SELECT * FROM reading_history ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_term_log(self, term: str, definition_zh: str = "") -> None:
        db = await self._get_db()
        cursor = db.execute("SELECT id, times_explained FROM term_log WHERE term = ?", (term,))
        row = cursor.fetchone()
        if row:
            db.execute("UPDATE term_log SET times_explained = ?, last_seen = CURRENT_TIMESTAMP WHERE id = ?", (row["times_explained"] + 1, row["id"]))
        else:
            db.execute("INSERT INTO term_log (term, definition_zh) VALUES (?, ?)", (term, definition_zh))
        db.commit()

    async def get_term_log(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = db.execute("SELECT * FROM term_log ORDER BY last_seen DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_session_summary(self, doc_id: str, topics: str = "", difficulty: str = "intermediate") -> None:
        db = await self._get_db()
        db.execute("INSERT INTO session_summaries (doc_id, topics, difficulty) VALUES (?, ?, ?)", (doc_id, topics, difficulty))
        db.commit()

    async def get_session_summaries(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = db.execute("SELECT * FROM session_summaries ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_validation_record(self, explanation_id: str, passed: bool, issues: list[str] | None = None) -> None:
        db = await self._get_db()
        issues_json = json.dumps(issues or [])
        db.execute("INSERT INTO validation_records (explanation_id, passed, issues) VALUES (?, ?, ?)", (explanation_id, 1 if passed else 0, issues_json))
        db.commit()

    async def get_validation_records(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = db.execute("SELECT * FROM validation_records ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_learning_event(
        self,
        event_type: str,
        doc_id: str = "",
        section_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        db = await self._get_db()
        db.execute(
            "INSERT INTO learning_events (event_type, doc_id, section_id, payload) VALUES (?, ?, ?, ?)",
            (event_type, doc_id, section_id, json.dumps(payload or {}, ensure_ascii=False)),
        )
        db.commit()

    async def get_learning_events(self, doc_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        db = await self._get_db()
        if doc_id:
            cursor = db.execute(
                "SELECT * FROM learning_events WHERE doc_id = ? ORDER BY id DESC LIMIT ?",
                (doc_id, limit),
            )
        else:
            cursor = db.execute(
                "SELECT * FROM learning_events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        rows = cursor.fetchall()
        events = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload") or "{}")
            except json.JSONDecodeError:
                item["payload"] = {}
            events.append(item)
        return events

    async def upsert_concept_memory(
        self,
        concept: str,
        doc_id: str = "",
        status: str = "learning",
        signal: str = "",
        section_id: str = "",
    ) -> None:
        concept = concept.strip()
        if not concept:
            return
        db = await self._get_db()
        cursor = db.execute(
            "SELECT id, evidence_count, source_events FROM concept_memory WHERE concept = ? AND doc_id = ?",
            (concept, doc_id),
        )
        row = cursor.fetchone()
        if row:
            try:
                source_events = json.loads(row["source_events"] or "[]")
            except json.JSONDecodeError:
                source_events = []
            if signal:
                source_events.append(signal)
            db.execute(
                """
                UPDATE concept_memory
                SET status = ?,
                    evidence_count = ?,
                    source_events = ?,
                    last_section_id = ?,
                    last_seen = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    status,
                    int(row["evidence_count"]) + 1,
                    json.dumps(source_events[-20:], ensure_ascii=False),
                    section_id,
                    row["id"],
                ),
            )
        else:
            db.execute(
                """
                INSERT INTO concept_memory
                    (concept, doc_id, status, evidence_count, source_events, last_section_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    concept,
                    doc_id,
                    status,
                    1,
                    json.dumps([signal] if signal else [], ensure_ascii=False),
                    section_id,
                ),
            )
        db.commit()

    async def get_concept_memory(self, doc_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        db = await self._get_db()
        if doc_id:
            cursor = db.execute(
                "SELECT * FROM concept_memory WHERE doc_id = ? ORDER BY last_seen DESC LIMIT ?",
                (doc_id, limit),
            )
        else:
            cursor = db.execute(
                "SELECT * FROM concept_memory ORDER BY last_seen DESC LIMIT ?",
                (limit,),
            )
        rows = cursor.fetchall()
        concepts = []
        for row in rows:
            item = dict(row)
            try:
                item["source_events"] = json.loads(item.get("source_events") or "[]")
            except json.JSONDecodeError:
                item["source_events"] = []
            concepts.append(item)
        return concepts

    async def clear_session_memory(self) -> None:
        db = await self._get_db()
        db.execute("DELETE FROM learning_events")
        db.commit()

    async def clear_document_memory(self, doc_id: str) -> None:
        db = await self._get_db()
        db.execute("DELETE FROM learning_events WHERE doc_id = ?", (doc_id,))
        db.execute("DELETE FROM concept_memory WHERE doc_id = ?", (doc_id,))
        db.execute("DELETE FROM reading_history WHERE doc_id = ?", (doc_id,))
        db.execute("DELETE FROM session_summaries WHERE doc_id = ?", (doc_id,))
        db.commit()

    async def clear_all_memory(self) -> None:
        db = await self._get_db()
        for table in (
            "learning_events",
            "concept_memory",
            "reading_history",
            "term_log",
            "session_summaries",
            "validation_records",
        ):
            db.execute(f"DELETE FROM {table}")
        db.commit()

    async def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
