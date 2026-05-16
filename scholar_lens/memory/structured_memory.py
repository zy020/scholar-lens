from __future__ import annotations

import json
import logging
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
    """Tier 2: Structured long-term memory stored in SQLite."""

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

    async def add_reading_record(self, doc_id: str, section_id: str, comprehension_score: float = 0.0) -> None:
        db = await self._get_db()
        await db.execute("INSERT INTO reading_history (doc_id, section_id, comprehension_score) VALUES (?, ?, ?)", (doc_id, section_id, comprehension_score))
        await db.commit()

    async def get_reading_history(self, doc_id: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM reading_history WHERE doc_id = ? ORDER BY timestamp DESC", (doc_id,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_all_reading_history(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM reading_history ORDER BY timestamp DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_term_log(self, term: str, definition_zh: str = "") -> None:
        db = await self._get_db()
        cursor = await db.execute("SELECT id, times_explained FROM term_log WHERE term = ?", (term,))
        row = await cursor.fetchone()
        if row:
            await db.execute("UPDATE term_log SET times_explained = ?, last_seen = CURRENT_TIMESTAMP WHERE id = ?", (row["times_explained"] + 1, row["id"]))
        else:
            await db.execute("INSERT INTO term_log (term, definition_zh) VALUES (?, ?)", (term, definition_zh))
        await db.commit()

    async def get_term_log(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM term_log ORDER BY last_seen DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_session_summary(self, doc_id: str, topics: str = "", difficulty: str = "intermediate") -> None:
        db = await self._get_db()
        await db.execute("INSERT INTO session_summaries (doc_id, topics, difficulty) VALUES (?, ?, ?)", (doc_id, topics, difficulty))
        await db.commit()

    async def get_session_summaries(self) -> list[dict[str, Any]]:
        db = await self._get_db()
        cursor = await db.execute("SELECT * FROM session_summaries ORDER BY timestamp DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def add_validation_record(self, explanation_id: str, passed: bool, issues: list[str] | None = None) -> None:
        db = await self._get_db()
        issues_json = json.dumps(issues or [])
        await db.execute("INSERT INTO validation_records (explanation_id, passed, issues) VALUES (?, ?, ?)", (explanation_id, 1 if passed else 0, issues_json))
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
