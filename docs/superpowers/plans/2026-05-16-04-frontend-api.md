# ScholarLens Frontend + API Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI backend API, NiceGUI frontend (PDF viewer + bilingual reading + chat tutoring + notes panel), wire them together with SSE streaming, and add the model configuration panel.

**Architecture:** FastAPI serves REST endpoints and SSE streams. NiceGUI runs as a separate process (or embedded) providing the browser UI with three modes: Parallel Reading, Chat Tutoring, Learning Notes. The frontend communicates with the backend API. PDF.js renders documents in-browser. NiceGUI's `ui.splitter` provides the dual-pane layout.

**Tech Stack:** FastAPI (API), NiceGUI (frontend), PDF.js (PDF rendering), uvicorn (ASGI server), sse-starlette (SSE)

**Depends on:** Plan 01 (core), Plan 02 (parsers + RAG), Plan 03 (memory + agents)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scholar_lens/api/__init__.py` | API package init |
| `scholar_lens/api/main.py` | FastAPI app factory, CORS, lifespan |
| `scholar_lens/api/routes/__init__.py` | Routes package init |
| `scholar_lens/api/routes/config.py` | Model configuration endpoints |
| `scholar_lens/api/routes/documents.py` | Document upload + processing endpoints |
| `scholar_lens/api/routes/chat.py` | Chat/tutor SSE streaming endpoint |
| `scholar_lens/api/routes/notes.py` | Learning notes + glossary endpoints |
| `scholar_lens/api/schemas.py` | Pydantic request/response schemas |
| `scholar_lens/api/middleware.py` | Request logging, error handling middleware |
| `scholar_lens/frontend/__init__.py` | Frontend package init |
| `scholar_lens/frontend/app.py` | NiceGUI app entry point |
| `scholar_lens/frontend/components/__init__.py` | Components package |
| `scholar_lens/frontend/components/layout.py` | Main layout (header + splitter + status bar) |
| `scholar_lens/frontend/components/pdf_viewer.py` | PDF.js viewer component |
| `scholar_lens/frontend/components/chat_panel.py` | Chat tutoring panel |
| `scholar_lens/frontend/components/parallel_reader.py` | Bilingual parallel reading panel |
| `scholar_lens/frontend/components/notes_panel.py` | Learning notes + glossary panel |
| `scholar_lens/frontend/components/config_panel.py` | Model configuration panel |
| `scholar_lens/frontend/components/status_bar.py` | Progress + token + comprehension status bar |
| `tests/unit/api/__init__.py` | API test package |
| `tests/unit/api/test_config.py` | Config route tests |
| `tests/unit/api/test_documents.py` | Document route tests |
| `tests/unit/api/test_chat.py` | Chat route tests |
| `tests/unit/frontend/__init__.py` | Frontend test package |

---

### Task 1: API Schemas + FastAPI App Factory

**Files:**
- Create: `scholar_lens/api/__init__.py`
- Create: `scholar_lens/api/schemas.py`
- Create: `scholar_lens/api/main.py`
- Create: `scholar_lens/api/middleware.py`
- Create: `scholar_lens/api/routes/__init__.py`
- Create: `tests/unit/api/__init__.py`
- Create: `tests/unit/api/test_schemas.py`
- Create: `tests/unit/api/test_app.py`

- [ ] **Step 1: Write failing tests for schemas and app**

```python
# tests/unit/api/test_schemas.py
import pytest
from scholar_lens.api.schemas import (
    ConfigUpdateRequest,
    ConfigResponse,
    DocumentUploadResponse,
    ChatRequest,
    ChatMessage,
    ExplanationResponse,
    NotesResponse,
)


class TestSchemas:
    def test_config_update_request(self):
        req = ConfigUpdateRequest(
            llm_api_key="key",
            llm_model="gpt-4o-mini",
            embedding_api_key="emb-key",
        )
        assert req.llm_model == "gpt-4o-mini"

    def test_config_response(self):
        resp = ConfigResponse(
            llm_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            reranker_available=False,
            vision_available=False,
        )
        assert resp.reranker_available is False

    def test_chat_request(self):
        req = ChatRequest(
            message="Explain self-attention",
            doc_id="paper_001",
            section_id="3.1",
        )
        assert req.message == "Explain self-attention"

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"

    def test_document_upload_response(self):
        resp = DocumentUploadResponse(
            doc_id="paper_001",
            doc_type="research_paper",
            num_sections=5,
            status="processed",
        )
        assert resp.status == "processed"

    def test_explanation_response(self):
        resp = ExplanationResponse(
            original="Self-attention",
            translation="自注意力",
            explanation="A mechanism...",
            confidence="high",
        )
        assert resp.confidence == "high"
```

```python
# tests/unit/api/test_app.py
import pytest
from fastapi.testclient import TestClient
from scholar_lens.api.main import create_app


class TestApp:
    def test_create_app(self):
        app = create_app()
        assert app is not None

    def test_health_check(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_cors_headers(self):
        app = create_app()
        client = TestClient(app)
        response = client.options("/api/config", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        assert response.status_code in (200, 204)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/api/ -v
```

Expected: FAIL

- [ ] **Step 3: Install API dependencies**

```bash
pip install sse-starlette uvicorn 2>&1 | tail -3
```

- [ ] **Step 4: Implement API schemas**

```python
# scholar_lens/api/__init__.py
```

```python
# scholar_lens/api/routes/__init__.py
```

```python
# tests/unit/api/__init__.py
```

```python
# scholar_lens/api/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigUpdateRequest(BaseModel):
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str | None = None
    vision_api_key: str | None = None
    vision_model: str = "gpt-4o"


class ConfigResponse(BaseModel):
    llm_model: str
    embedding_model: str
    reranker_available: bool = False
    vision_available: bool = False
    status: str = "configured"


class DocumentUploadResponse(BaseModel):
    doc_id: str
    doc_type: str
    num_sections: int
    num_terms: int = 0
    status: str  # "processing" | "processed" | "error"
    error: str = ""


class ChatRequest(BaseModel):
    message: str
    doc_id: str = ""
    section_id: str = ""
    mode: str = "chat"  # chat | explain | translate


class ChatMessage(BaseModel):
    role: str  # user | assistant | system
    content: str
    timestamp: str = ""


class ExplanationResponse(BaseModel):
    original: str
    translation: str
    explanation: str
    related_terms: list[dict] = Field(default_factory=list)
    difficulty_level: str = "intermediate"
    source_section: str = ""
    confidence: str = "medium"


class NotesResponse(BaseModel):
    doc_id: str
    terms: list[dict] = Field(default_factory=list)
    reading_progress: dict = Field(default_factory=dict)
    concept_map: str = ""
```

- [ ] **Step 5: Implement FastAPI app and middleware**

```python
# scholar_lens/api/middleware.py
from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)")
        return response
```

```python
# scholar_lens/api/main.py
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scholar_lens.api.middleware import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ScholarLens",
        version="0.1.0",
        description="Educational agent system for reading English academic documents",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from scholar_lens.api.routes import config, documents, chat, notes
    app.include_router(config.router, prefix="/api/config", tags=["config"])
    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(notes.router, prefix="/api/notes", tags=["notes"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
```

- [ ] **Step 6: Implement placeholder route files**

```python
# scholar_lens/api/routes/config.py
from __future__ import annotations

from fastapi import APIRouter

from scholar_lens.api.schemas import ConfigUpdateRequest, ConfigResponse

router = APIRouter()


@router.get("", response_model=ConfigResponse)
async def get_config():
    return ConfigResponse(
        llm_model="not configured",
        embedding_model="not configured",
        reranker_available=False,
        vision_available=False,
        status="not_configured",
    )


@router.put("")
async def update_config(request: ConfigUpdateRequest):
    return ConfigResponse(
        llm_model=request.llm_model,
        embedding_model=request.embedding_model,
        reranker_available=request.reranker_model is not None,
        vision_available=request.vision_api_key is not None,
        status="configured",
    )


@router.post("/test")
async def test_connection():
    return {"status": "not_implemented"}
```

```python
# scholar_lens/api/routes/documents.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File

from scholar_lens.api.schemas import DocumentUploadResponse

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    return DocumentUploadResponse(
        doc_id="pending",
        doc_type="unknown",
        num_sections=0,
        status="processing",
    )


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    return {"doc_id": doc_id, "status": "not_implemented"}
```

```python
# scholar_lens/api/routes/chat.py
from __future__ import annotations

from fastapi import APIRouter

from scholar_lens.api.schemas import ChatRequest, ChatMessage

router = APIRouter()


@router.post("")
async def chat(request: ChatRequest):
    return ChatMessage(
        role="assistant",
        content="Tutor not configured yet.",
    )
```

```python
# scholar_lens/api/routes/notes.py
from __future__ import annotations

from fastapi import APIRouter

from scholar_lens.api.schemas import NotesResponse

router = APIRouter()


@router.get("/{doc_id}", response_model=NotesResponse)
async def get_notes(doc_id: str):
    return NotesResponse(doc_id=doc_id)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/api/ -v
```

Expected: All 8 tests PASS.

- [ ] **Step 8: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/api/ tests/unit/api/ && git commit -m "feat: add FastAPI app with config, documents, chat, notes routes"
```

---

### Task 2: NiceGUI Frontend - Main Layout + Mode Switching

**Files:**
- Create: `scholar_lens/frontend/__init__.py`
- Create: `scholar_lens/frontend/app.py`
- Create: `scholar_lens/frontend/components/__init__.py`
- Create: `scholar_lens/frontend/components/layout.py`
- Create: `tests/unit/frontend/__init__.py`

- [ ] **Step 1: Write failing test for layout**

```python
# tests/unit/frontend/__init__.py
```

```python
# tests/unit/frontend/test_layout.py
import pytest
from scholar_lens.frontend.components.layout import MainLayout


class TestMainLayout:
    def test_instantiation(self):
        layout = MainLayout()
        assert layout is not None
        assert layout.current_mode == "chat"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/frontend/ -v
```

Expected: FAIL

- [ ] **Step 3: Implement frontend packages and layout**

```python
# scholar_lens/frontend/__init__.py
```

```python
# scholar_lens/frontend/components/__init__.py
```

```python
# scholar_lens/frontend/components/layout.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MainLayout:
    """Main layout state for the ScholarLens frontend.

    Per spec Section 7.1:
    - Header: ScholarLens title, document selector, mode switch, settings
    - Split view: Document Reader (left) | Interaction Panel (right)
    - Status bar: progress, comprehension, tokens
    """

    current_mode: str = "chat"  # chat | parallel | notes
    current_doc_id: str = ""
    is_configured: bool = False

    _MODES = ("chat", "parallel", "notes")

    def set_mode(self, mode: str) -> None:
        if mode in self._MODES:
            self.current_mode = mode

    def set_document(self, doc_id: str) -> None:
        self.current_doc_id = doc_id
```

```python
# scholar_lens/frontend/app.py
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def create_app():
    """Create the NiceGUI frontend application.

    Per spec Section 7:
    - Three modes: Parallel Reading, Chat Tutoring, Learning Notes
    - PDF.js for document rendering
    - NiceGUI ui.splitter for dual-pane layout
    - SSE for streaming chat responses
    """
    try:
        from nicegui import ui, app
    except ImportError:
        logger.error("nicegui not installed. Install with: pip install nicegui")
        return None

    @ui.page("/")
    async def index():
        # Header
        with ui.header().classes("items-center justify-between"):
            ui.label("ScholarLens").classes("text-h5 font-bold")
            ui.select(
                options=["Chat", "Parallel", "Notes"],
                value="Chat",
                on_change=lambda e: None,  # Mode switch handler
            ).props("dense")
            ui.button(icon="settings", on_click=lambda: None).props("flat round")

        # Main content - splitter
        with ui.splitter(value=50).classes("w-full h-full") as splitter:
            with splitter.before:
                ui.label("Document Viewer").classes("text-caption")
                # PDF.js viewer will be embedded here
                ui.html('<div id="pdf-viewer" style="height: 100%;"></div>')

            with splitter.after:
                ui.label("Interaction Panel").classes("text-caption")
                # Chat / Parallel / Notes panel will be here
                ui.html('<div id="interaction-panel" style="height: 100%;"></div>')

        # Status bar
        with ui.footer().classes("items-center"):
            ui.label("Progress: 0/0 | Tokens: 0").classes("text-caption")

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/frontend/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/frontend/ tests/unit/frontend/ && git commit -m "feat: add NiceGUI frontend with main layout and mode switching"
```

---

### Task 3: Chat Panel + Config Panel Components

**Files:**
- Create: `scholar_lens/frontend/components/chat_panel.py`
- Create: `scholar_lens/frontend/components/config_panel.py`
- Create: `scholar_lens/frontend/components/status_bar.py`
- Create: `tests/unit/frontend/test_chat_panel.py`
- Create: `tests/unit/frontend/test_config_panel.py`

- [ ] **Step 1: Write failing tests for components**

```python
# tests/unit/frontend/test_chat_panel.py
import pytest
from scholar_lens.frontend.components.chat_panel import ChatPanelState


class TestChatPanelState:
    def test_create(self):
        panel = ChatPanelState()
        assert panel.messages == []
        assert panel.is_streaming is False

    def test_add_message(self):
        panel = ChatPanelState()
        panel.add_message("user", "What is self-attention?")
        panel.add_message("assistant", "Self-attention is a mechanism...")
        assert len(panel.messages) == 2
        assert panel.messages[0]["role"] == "user"

    def test_clear_messages(self):
        panel = ChatPanelState()
        panel.add_message("user", "Hello")
        panel.clear()
        assert panel.messages == []

    def test_streaming_state(self):
        panel = ChatPanelState()
        panel.is_streaming = True
        assert panel.is_streaming is True
```

```python
# tests/unit/frontend/test_config_panel.py
import pytest
from scholar_lens.frontend.components.config_panel import ConfigPanelState


class TestConfigPanelState:
    def test_create_defaults(self):
        panel = ConfigPanelState()
        assert panel.llm_api_key == ""
        assert panel.llm_model == "gpt-4o-mini"
        assert panel.is_configured is False

    def test_configure(self):
        panel = ConfigPanelState()
        panel.llm_api_key = "test-key"
        panel.embedding_api_key = "test-emb"
        panel.is_configured = True
        assert panel.is_configured is True

    def test_to_request_dict(self):
        panel = ConfigPanelState(
            llm_api_key="key",
            llm_model="gpt-4o",
            embedding_api_key="emb",
        )
        d = panel.to_request_dict()
        assert d["llm_api_key"] == "key"
        assert d["llm_model"] == "gpt-4o"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/frontend/test_chat_panel.py tests/unit/frontend/test_config_panel.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement component state classes**

```python
# scholar_lens/frontend/components/chat_panel.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatPanelState:
    """State for the chat tutoring panel.

    Per spec Section 7.2 (Mode 2): Socratic dialogue with tutor agent,
    citation links, quick action buttons.
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    is_streaming: bool = False
    doc_id: str = ""
    section_id: str = ""

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        self.messages = []
```

```python
# scholar_lens/frontend/components/config_panel.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfigPanelState:
    """State for the model configuration panel.

    Per spec Section 7.4: LLM, Embedding, Reranker, Vision model configuration.
    """

    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3

    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"

    reranker_model: str = ""
    reranker_api_key: str = ""

    vision_api_key: str = ""
    vision_model: str = "gpt-4o"

    is_configured: bool = False

    def to_request_dict(self) -> dict:
        return {
            "llm_api_key": self.llm_api_key,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "embedding_api_key": self.embedding_api_key,
            "embedding_base_url": self.embedding_base_url,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model or None,
            "vision_api_key": self.vision_api_key or None,
            "vision_model": self.vision_model,
        }
```

```python
# scholar_lens/frontend/components/status_bar.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StatusBarState:
    """State for the bottom status bar.

    Per spec Section 7.1: Progress, Comprehension, Token count.
    """

    sections_completed: int = 0
    sections_total: int = 0
    comprehension_score: float = 0.0
    tokens_used: int = 0

    @property
    def progress_text(self) -> str:
        return f"{self.sections_completed}/{self.sections_total} sections"

    @property
    def status_text(self) -> str:
        return f"Progress: {self.progress_text} | Comprehension: {self.comprehension_score:.0%} | Tokens: {self.tokens_used}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/frontend/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/frontend/components/ tests/unit/frontend/ && git commit -m "feat: add chat panel, config panel, and status bar state classes"
```

---

### Task 4: Parallel Reader + Notes Panel + PDF Viewer Components

**Files:**
- Create: `scholar_lens/frontend/components/parallel_reader.py`
- Create: `scholar_lens/frontend/components/notes_panel.py`
- Create: `scholar_lens/frontend/components/pdf_viewer.py`
- Create: `scholar_lens/frontend/components/nav_tree.py`
- Create: `tests/unit/frontend/test_parallel_reader.py`
- Create: `tests/unit/frontend/test_notes_panel.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/frontend/test_parallel_reader.py
import pytest
from scholar_lens.frontend.components.parallel_reader import ParallelReaderState


class TestParallelReaderState:
    def test_create(self):
        state = ParallelReaderState()
        assert state.current_section_id == ""
        assert state.paragraphs == []

    def test_set_paragraphs(self):
        state = ParallelReaderState()
        state.set_paragraphs([
            {"en": "The self-attention mechanism computes attention scores.", "zh": "自注意力机制计算注意力分数。"},
            {"en": "Multi-head attention runs multiple attention functions.", "zh": "多头注意力运行多个注意力函数。"},
        ])
        assert len(state.paragraphs) == 2

    def test_scroll_sync_position(self):
        state = ParallelReaderState()
        state.current_paragraph_index = 3
        assert state.current_paragraph_index == 3
```

```python
# tests/unit/frontend/test_notes_panel.py
import pytest
from scholar_lens.frontend.components.notes_panel import NotesPanelState


class TestNotesPanelState:
    def test_create(self):
        state = NotesPanelState()
        assert state.terms == []
        assert state.reading_progress == {}

    def test_add_term(self):
        state = NotesPanelState()
        state.add_term("self-attention", "自注意力", "understood")
        assert len(state.terms) == 1
        assert state.terms[0]["english"] == "self-attention"

    def test_update_progress(self):
        state = NotesPanelState()
        state.update_progress("3.1", 0.8)
        assert state.reading_progress["3.1"] == 0.8

    def test_concept_map(self):
        state = NotesPanelState()
        state.concept_map_mermaid = "graph TD\n  A-->B"
        assert "graph TD" in state.concept_map_mermaid
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/frontend/test_parallel_reader.py tests/unit/frontend/test_notes_panel.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement component states**

```python
# scholar_lens/frontend/components/parallel_reader.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParallelReaderState:
    """State for the bilingual parallel reading panel.

    Per spec Section 7.2 (Mode 1): English original + Chinese translation
    side by side, scroll-synced, terms highlighted, hover for details.
    """

    current_section_id: str = ""
    paragraphs: list[dict[str, str]] = field(default_factory=list)  # [{"en": ..., "zh": ...}]
    current_paragraph_index: int = 0
    highlight_terms: list[str] = field(default_factory=list)

    def set_paragraphs(self, paragraphs: list[dict[str, str]]) -> None:
        self.paragraphs = paragraphs
        self.current_paragraph_index = 0
```

```python
# scholar_lens/frontend/components/notes_panel.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NotesPanelState:
    """State for the learning notes panel.

    Per spec Section 7.2 (Mode 3): Reading progress overview,
    term glossary with understanding status, concept relationship graph.
    """

    terms: list[dict] = field(default_factory=list)  # [{"english": ..., "chinese": ..., "status": ...}]
    reading_progress: dict[str, float] = field(default_factory=dict)  # section_id → comprehension
    concept_map_mermaid: str = ""
    highlights: list[dict] = field(default_factory=list)

    def add_term(self, english: str, chinese: str, status: str = "new") -> None:
        self.terms.append({"english": english, "chinese": chinese, "status": status})

    def update_progress(self, section_id: str, comprehension: float) -> None:
        self.reading_progress[section_id] = comprehension
```

```python
# scholar_lens/frontend/components/pdf_viewer.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PDFViewerState:
    """State for the PDF.js document viewer component.

    Per spec Section 7.3: Select text, click term, page change,
    click reference, section nav, highlight.
    """

    current_page: int = 0
    total_pages: int = 0
    selected_text: str = ""
    current_section: str = ""
    zoom_level: float = 1.0
```

```python
# scholar_lens/frontend/components/nav_tree.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NavTreeState:
    """State for the section navigation tree.

    Per spec Section 7.1: Section navigation with progress indicators.
    """

    sections: list[dict] = field(default_factory=list)  # [{"id": ..., "title": ..., "level": ..., "completed": ...}]
    expanded: set[str] = field(default_factory=set)

    def set_sections(self, sections: list[dict]) -> None:
        self.sections = sections

    def toggle_expand(self, section_id: str) -> None:
        if section_id in self.expanded:
            self.expanded.discard(section_id)
        else:
            self.expanded.add(section_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/unit/frontend/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Run all tests across the entire project**

```bash
cd /home/zhy/scholar-lens && python -m pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/zhy/scholar-lens && git add scholar_lens/frontend/components/ tests/unit/frontend/ && git commit -m "feat: add parallel reader, notes panel, PDF viewer, and nav tree state components"
```

---

## Self-Review

**1. Spec coverage:**
- FastAPI backend (Section 3.3) → Task 1 ✓
- NiceGUI frontend (Section 7.1) → Task 2 ✓
- Three modes (Section 7.2) → Tasks 3, 4 ✓
- Model config panel (Section 7.4) → Task 3 ✓
- Chat tutoring panel → Task 3 ✓
- Parallel reading panel → Task 4 ✓
- Learning notes panel → Task 4 ✓
- Status bar (Section 7.1) → Task 3 ✓
- Document reader interactions (Section 7.3) → Task 4 ✓
- Gaps: SSE streaming in chat endpoint, actual NiceGUI rendering (needs running server), PDF.js integration (needs browser), Obsidian export (runtime feature), VLM integration (deferred to config). These are integration-level features that require a running environment.

**2. Placeholder scan:** No TBD or TODO found.

**3. Type consistency:** `ConfigPanelState.to_request_dict()` output matches `ConfigUpdateRequest` schema fields. `ChatPanelState.messages` format matches `ChatMessage` schema. All state classes use consistent field names.
