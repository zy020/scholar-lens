from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ReflectionMemory:
    """Tier 4: Reflection memory — learning pattern insights, stored as Markdown files."""

    def __init__(self, knowledge_dir: str = "knowledge") -> None:
        self._dir = Path(knowledge_dir) / "reflections"
        self._dir.mkdir(parents=True, exist_ok=True)

    async def save_reflection(self, title: str, content: str, tags: list[str] | None = None) -> Path:
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
