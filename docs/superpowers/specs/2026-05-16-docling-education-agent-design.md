# ScholarLens: Educational Agent System for Reading English Academic Documents

> Design Spec — 2026-05-16

## 1. Problem Statement

Chinese university students frequently need to read English academic papers and courseware. For non-native speakers, language barriers significantly reduce reading efficiency and comprehension. Existing tools either provide pure translation (losing conceptual context) or generic Q&A (without pedagogical scaffolding). No system integrates bilingual reading assistance, conceptual understanding, and adaptive tutoring in a single agent workflow.

## 2. Target User

- **Primary user**: The developer (single-user, local deployment)
- **Secondary users**: Open-source community (GitHub release)
- **Deployment**: Local machine, personal use, self-configured API keys
- **Context**: Chinese-native university students reading English research papers and courseware

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NiceGUI Frontend                      │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ PDF Viewer│  │ Bilingual/   │  │  Chat Tutoring   │  │
│  │ (PDF.js) │  │ Notes Panel  │  │  (Socratic)      │  │
│  └─────┬────┘  └──────┬───────┘  └────────┬─────────┘  │
│        └──────────────┼───────────────────┘             │
│                       │ SSE + REST                      │
└───────────────────────┼─────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────┐
│              FastAPI Backend                             │
│                       │                                 │
│  ┌────────────────────┼────────────────────────────┐    │
│  │         LangGraph Orchestration Layer             │    │
│  │                                                    │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────┐ │    │
│  │  │  Document   │→ │  Content     │→ │Validator │ │    │
│  │  │  Analyzer   │  │  Explainer   │  │  Agent   │ │    │
│  │  └──────┬──────┘  └──────┬───────┘  └────┬────┘ │    │
│  │         │                │               │       │    │
│  │         └────────────────┼───────────────┘       │    │
│  │                          ↓                       │    │
│  │                   ┌────────────┐                 │    │
│  │                   │  Learning  │                 │    │
│  │                   │   Tutor    │                 │    │
│  │                   │   Agent    │                 │    │
│  │                   └────────────┘                 │    │
│  │                          │                       │    │
│  │              ┌───────────┴───────────┐           │    │
│  │              │  Four-Tier Memory     │           │    │
│  │              │  System               │           │    │
│  │              └───────────────────────┘           │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │               Infrastructure Layer                │    │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────────────┐ │    │
│  │  │ Parsing  │ │   RAG    │ │  Obsidian Knowledge│ │    │
│  │  │ Pipeline │ │ Pipeline │ │  Base Output       │ │    │
│  │  └──────────┘ └──────────┘ └───────────────────┘ │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Design Principles

1. **Lazy explanation** — Generate explanations on-demand as the student reads, not pre-computed for the entire document
2. **Layered context loading** — L0/L1/L2 three-tier document content, loaded on demand to save tokens
3. **Closed-loop learning** — Read → Feedback → Update student model → Adjust strategy, improving over time
4. **White-box traceability** — All memory and retrieval paths are traceable and debuggable (inspired by OpenViking/TencentDB)
5. **Portable knowledge** — Output as Obsidian-compatible Markdown; the student owns their knowledge base
6. **Graceful degradation** — Every advanced capability (VLM, reranker, Nougat) has a fallback; the system works with minimum config (LLM + embedding)

### 3.3 Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Orchestration | LangGraph | Pipeline + interactive hybrid, typed state, streaming, checkpointing |
| Parsing | Docling + Nougat + GROBID | PDF+PPT unified, best formula extraction, citation analysis |
| Embedding/Retrieval | BGE-M3 + Contextual Retrieval + CRAG | EN-ZH cross-lingual, high retrieval quality |
| Reranking | bge-reranker-m3 + rule fallback | Model-based when available, rule-based when not |
| Vector Store | ChromaDB | Lightweight, Python-native |
| Memory Storage | SQLite + Markdown files | Local-first, human-readable, debuggable |
| Backend | FastAPI | SSE streaming, async, proven |
| Frontend | NiceGUI + PDF.js | Python-native, splitter+chat, Vue.js extensible |
| Knowledge Output | Obsidian-compatible Markdown | Human-readable, human-editable, portable |

## 4. Four Agents

### 4.1 Document Analyzer Agent

**Responsibility**: Parse document, extract structure, concepts, difficulty; generate Mermaid structure map and L0/L1/L2 layered content

**Input**: Raw PDF/PPT file
**Output**: Structured `DocumentUnderstanding` object

```python
class DocumentUnderstanding:
    doc_type: str              # "research_paper" | "courseware" | "textbook_chapter"
    language: str              # "en" | "zh" | "mixed"
    difficulty: str            # "beginner" | "intermediate" | "advanced"
    estimated_reading_time: int

    # Structure
    sections: list[Section]
    mermaid_map: str           # Mermaid paper structure diagram (injected into agent context)

    # Concepts
    key_terms: list[Term]      # Key terms (EN + ZH mapping)
    prerequisites: list[str]   # Prerequisite knowledge

    # Layered content
    l0_summaries: dict[str, str]  # section_id → ~100 token summary
    l1_overviews: dict[str, str]  # section_id → ~2k token overview
    # l2 = raw chunks, stored in RAG pipeline

    # References (separated)
    references: list[Reference]   # Structured reference metadata
    citation_contexts: list[CitationContext]  # In-text citation contexts (high value)
```

**Processing flow**:
```
PDF/PPT → Docling parsing → Nougat formula enhancement
                                    ↓
                    LLM: identify doc type + language + difficulty
                                    ↓
                    LLM: extract section structure → generate Mermaid map
                                    ↓
                    LLM: extract key terms + prerequisites (SciERC relation types)
                                    ↓
                    LLM: generate L0/L1/L2 layered content
                                    ↓
                    GROBID: extract citations + reference metadata (separated)
                                    ↓
                    Output DocumentUnderstanding + write to RAG index
```

**Key design decisions**:
- Mermaid diagram serves as "paper map" injected into subsequent agent context, replacing full-text loading
- Term extraction uses SciERC relation types (Used-for, Hyponym-of, Part-of) to build prerequisite chains
- L0/L1 generated at upload time; L2 (raw content) stored in vector store for on-demand retrieval
- References section separated: reference list stored as structured metadata, NOT indexed in vector store; citation contexts ("We extend [3]") preserved as normal chunks

### 4.2 Content Explainer Agent

**Responsibility**: On-demand translation, term explanation, sentence breakdown, concept connection

**Core characteristic**: Lazy loading, called on demand

**Input**: Student request + DocumentUnderstanding + student model
**Output**: Bilingual explanation content

```python
class ExplanationRequest:
    section_id: str
    mode: str           # "translate" | "explain" | "term_lookup" | "sentence_breakdown"
    target_text: str     # Student-selected text (optional)
    student_level: str   # From student model
    previous_explanations: list  # Already given in this session

class ExplanationResult:
    original: str              # Original English
    translation: str           # Chinese translation (preserving key English terms)
    explanation: str           # Detailed Chinese explanation
    related_terms: list[Term]  # Related terms
    difficulty_level: str      # Difficulty of this explanation
    source_section: str        # Citation source
    confidence: str            # "high" | "medium" | "low" | "unverified"
```

**Three invocation modes**:

| Mode | Trigger | Content |
|------|---------|---------|
| **Bilingual parallel** | Student switches to "parallel reading" mode | Per-paragraph English + Chinese, terms preserve English |
| **On-demand explain** | Student clicks/selects text | Translation + concept explanation + related terms |
| **Tutor request** | Tutor agent determines student needs explanation | Explanation adapted to student level (scaffolding) |

**Translation strategy**:
- Key terms preserved inline: `self-attention mechanism（自注意力机制）`
- Term consistency: maintain per-document `bilingual_glossary`, same term unified translation throughout
- Formulas preserved as LaTeX with Chinese meaning: `$\text{Attention}(Q,K,V) = ...$（注意力计算公式）`

### 4.3 Validator Agent

**Responsibility**: Cross-verify explainer output — term accuracy, faithfulness to source, hallucination detection

**Trigger conditions**:
- First-time term explanation
- Student marks explanation as "questionable"
- Advanced difficulty sections
- Probability sampling (10% of explanations, for ongoing quality monitoring)

**Validation methods**:

| Method | Cost | When Used |
|--------|------|-----------|
| Rule: term consistency | Zero | Always |
| Rule: source backtracking (L2 chunk match) | Zero | Always |
| LLM: hallucination detection | Low (cheap model) | Trigger conditions above |
| LLM: factual accuracy | Medium | Advanced sections, student-flagged |

**Output**:
```python
class ValidationResult:
    passed: bool
    confidence: str       # "high" | "medium" | "low"
    issues: list[str]     # Specific issues found
    correction: str | None  # Suggested correction if failed
```

**Failure handling**: Validation failed → Explainer regenerates with correction hint, max 1 retry. If still fails, mark as "low confidence" and present to student with caveat.

### 4.4 Learning Tutor Agent

**Responsibility**: Socratic guidance, scaffolding, knowledge tracing, session management

**This is the only agent that directly converses with the student.** Other agents serve through the tutor.

**Interaction modes**:

| Mode | Description | When Triggered |
|------|-------------|----------------|
| **Collaborative reading** | "Coming up is Chapter 3, watch for these key concepts..." | Student enters new section |
| **Socratic questioning** | "Why do you think self-attention works better than RNN here?" | Student claims understanding |
| **Scaffolding** | Level 1 (full explanation) → Level 2 (hints) → Level 3 (questions only) | Dynamic, based on student model |
| **Teach-back** | "Can you explain positional encoding in your own words?" | After section completion |
| **Gap detection** | "To understand this, you need the concept of attention first" | Student model detects prerequisite gap |

**Collaboration with other agents**:
```
Student question → Tutor Agent judges:
  ├─ General knowledge → Answer directly (no retrieval)
  ├─ Document content → Retrieve RAG → Answer with citations
  ├─ Needs deep explanation → Request Explainer Agent → Wrap and reply
  └─ Student struggling → Downgrade scaffolding → Request simpler explanation
```

**Knowledge tracing**:
- After each interaction, update `p(known)` for relevant concepts in student model
- Prerequisite graph propagation: understanding "self-attention" boosts "multi-head attention" `p(known)`
- Periodic reflection memory generation

### 4.5 Inter-Agent Data Flow

```
Document upload → Document Analyzer (one-time, generates DocumentUnderstanding)
                      ↓ writes to shared state
                      ↓ Mermaid map + L0/L1 summaries + term glossary

Student interaction → Learning Tutor (continuous dialogue)
                      ↓ reads Mermaid map (lightweight, replaces full text)
                      ↓ on-demand requests to Explainer
                      ↓                    ↓
                      ↓           Content Explainer
                      ↓           reads L0→judge→load L1/L2
                      ↓           reads term glossary (consistency)
                      ↓           reads student model (adapt difficulty)
                      ↓                    ↓
                      ↓           Validator Agent checks
                      ↓                    ↓
                      ← returns ExplanationResult ←
                      ↓
                 Generate reply + update student model + update memory
```

**Key**: Tutor agent context only contains Mermaid map + L0 summaries + student model, NOT full text. Details obtained through Explainer agent, keeping context lean.

## 5. Four-Tier Memory System

### 5.1 Architecture

Inspired by OpenViking (layered loading), TencentDB Agent Memory (semantic pyramid), Hermes (closed-loop learning):

```
┌─────────────────────────────────────────────────┐
│ Tier 1: Core Memory (always in context)         │
│ · Student profile (3-5 sentences)               │
│ · Current reading position (doc + section)      │
│ · Active term glossary (last 10-20 terms)       │
│ · Current session summary (1-2 paragraphs)      │
│ Storage: System prompt                           │
│ Token budget: ~500                               │
├─────────────────────────────────────────────────┤
│ Tier 2: Structured Long-term Memory             │
│ · Reading history: (doc_id, section, ts, score) │
│ · Term log: (term, definition, times_explained) │
│ · Session summaries: (date, doc, topics, diff)  │
│ · Validation records: (id, passed, issues)      │
│ Storage: SQLite (structured query) + Markdown    │
│ Retrieval: BM25 + metadata filtering            │
├─────────────────────────────────────────────────┤
│ Tier 3: Document Memory (RAG pipeline)          │
│ · L0: Section summaries (~100 token/section)    │
│ · L1: Section overviews (~2k token/section)     │
│ · L2: Raw chunks (vector store, loaded on demand)│
│ · Mermaid paper structure diagram                │
│ Storage: ChromaDB (vectors) + SQLite (L0/L1)    │
│ Retrieval: BGE-M3 cross-lingual + BM25 + RRF    │
├─────────────────────────────────────────────────┤
│ Tier 4: Reflection Memory                       │
│ · Learning pattern insights                     │
│ · Strategy adjustment suggestions               │
│ · Knowledge gap summaries                       │
│ Storage: Markdown files (human-readable)         │
│ Generation: every 5 reading sessions             │
└─────────────────────────────────────────────────┘
```

### 5.2 Agent-Memory Interaction

| Agent | Read T1 | Read T2 | Read T3 | Write T2 | Write T4 |
|-------|---------|---------|---------|----------|----------|
| Document Analyzer | - | - | Write L0/L1/L2 | Write term glossary | - |
| Content Explainer | Read student profile | Read term glossary | Read L0→L1→L2 | Write explanation record | - |
| Validator Agent | - | Read term glossary | Read L2 (backtrack) | Write validation record | - |
| Learning Tutor | Read all | Read all | Via Explainer | Write reading history | Write reflections |

### 5.3 Obsidian Knowledge Base Output

Tier 2 and Tier 4 Markdown files in Obsidian-compatible format:

```
~/.docling/knowledge/
├── student_profile.md
├── reading_log/
│   ├── 2026-05-16_transformer.md
│   └── 2026-05-17_bert.md
├── glossary/
│   └── transformer_terms.md
├── reflections/
│   └── 2026-05-20_reflection.md
└── concept_maps/
    └── transformer_concepts.md
```

Agent reads from and writes to this directory, creating a bidirectional knowledge interface.

### 5.4 Token Budget Control

```
Single interaction token budget:
┌──────────────────────────────────────┐
│ System prompt + Tier1 core    ~800   │
│ Mermaid paper structure map   ~300   │
│ Retrieval results (T3 L1/L2) ~1500  │
│ Conversation history (last 5) ~1000  │
│ User input                    ~200   │
│ ──────────────────────────────────── │
│ Input total                  ~3800   │
│ Output budget                ~800    │
│ ──────────────────────────────────── │
│ Total per interaction        ~4600   │
│                                      │
│ vs. flat full-text loading  ~15000+  │
│ Savings: ~70%                        │
└──────────────────────────────────────┘
```

## 6. RAG Pipeline & Document Parsing

### 6.1 End-to-End Processing

```
PDF/PPT upload
    ↓
1. Parsing Pipeline
   PDF → Docling (structure + tables + images)
       → Nougat (formula LaTeX enhancement)
       → GROBID (citation + reference extraction)
   PPT → Docling (text + notes + images)
       → python-pptx (speaker notes)
    ↓
2. Chunking Pipeline
   Section-aware chunking (respect document structure)
   Formula-aware (formula + explanation text never split)
   Cross-reference annotation (see Fig 3 / see Eq 5)
   section_type metadata
   References SEPARATED (not chunked, not indexed)
    ↓
3. Indexing Pipeline
   Contextual Retrieval prefix generation
   L0/L1/L2 layered summary generation
   BGE-M3 embedding + ChromaDB vector index
   BM25 index (Chinese + English tokenization)
   Bilingual glossary extraction
    ↓
4. Retrieval Pipeline (runtime)
   CRAG: retrieve → relevance check → decide whether to use
   Hybrid: BM25 + BGE-M3 + RRF fusion
   Layered loading: L0→judge→L1→judge→L2
   bge-reranker-m3 reranking (with rule fallback)
```

### 6.2 Chunking Strategy

| Content Type | Strategy | Chunk Size | Metadata |
|-------------|----------|-----------|----------|
| **Prose paragraphs** | Section boundary first, paragraph boundary within large sections | 300-800 token | section_type, chapter, difficulty_score |
| **Formula-dense regions** | Formula + preceding explanation + variable definitions as one chunk, never split | Unlimited (completeness first) | has_formula, formula_ids, referenced_terms |
| **Tables/figures** | Entire table as one chunk + context description | Unlimited | content_type="table"/"figure", caption, referenced_by |

**Cross-reference handling**: Each chunk's metadata includes `cross_refs`. When a chunk is retrieved, referenced chunks are also returned to ensure context completeness.

### 6.3 Reference Separation

```
Parsing stage:
  Docling/GROBID identifies → marks References section as section_type="references"

  Two-category processing:
  ┌─ Citation contexts (high value): "We extend [3]" surrounding text
  │  → Preserved as normal chunks, section_type="citation_context"
  │  → Normal retrieval weight
  │
  └─ Reference list (low value): [1] Vaswani, A., et al. Attention Is All...
     → NOT chunked, NOT indexed in vector store
     → Stored separately as references.json
     → Queried only when student asks "what does this paper cite"
```

### 6.4 PDF Subtype Detection

```python
def detect_pdf_subtype(pdf_path) -> str:
    """Distinguish research paper PDF from courseware PDF"""
    pages = extract_page_features(pdf_path, sample_n=5)
    avg_chars_per_page = mean(p.char_count for p in pages)
    has_two_column = any(p.is_two_column for p in pages)
    has_abstract = detect_abstract(pdf_path)

    if avg_chars_per_page < 300 and not has_two_column:
        return "slides_pdf"       # Courseware exported as PDF
    elif has_abstract or has_two_column:
        return "research_paper"   # Academic paper
    else:
        return "general_document"
```

| Feature | Research Paper PDF | Courseware PDF | PPTX |
|---------|-------------------|----------------|------|
| Parser | Docling + Nougat + GROBID | Docling + VLM | Docling + python-pptx |
| Chunking | Section-aware, dense | Per-page/per-slide, sparse | Per-slide |
| Formulas | Nougat enhanced | Rare, on-demand | Rare |
| Figures | VLM description | VLM description (critical) | VLM description (critical) |
| Speaker notes | N/A | N/A | python-pptx extraction |
| Structure | Abstract/Method/Results... | Slide title + key points | Same as courseware PDF |

### 6.5 Cross-Lingual Retrieval

```
Student asks in Chinese → BGE-M3 embedding (shared EN-ZH vector space)
                       → Directly retrieve English chunks
                       → Retrieve English chunk + L1 Chinese overview (pre-generated at index time)
                       → Explainer generates Chinese answer based on English source + Chinese overview
                       → Answer preserves key English terms inline
```

No query translation needed — BGE-M3 natively supports Chinese query retrieving English documents.

### 6.6 Reranking Strategy

```
Level 1: Dedicated reranker model (bge-reranker-m3 / Cohere Rerank)
  → Best quality, requires extra config or API
  → User can optionally configure

Level 2: LLM as reranker (RankGPT approach)
  → Uses user's configured LLM for listwise reranking
  → No extra config, but slow and expensive

Level 3: Rule reranking (no model)
  → Section_type relevance weighting
  → Difficulty vs student_level match weighting
  → Cross-ref density weighting
  → Zero cost, zero latency

Level 4: No reranking
  → Direct RRF fusion results
  → Baseline
```

Auto-selection: if user configures reranker model → Level 1; else if embedding API supports reranking → Level 1; else → Level 3 (rule-based).

### 6.7 Vision Model Degradation

```
Level 1: VLM configured (GPT-4o-vision / Claude / Qwen-VL)
  → Detailed figure/chart descriptions
  → Full courseware PDF slide understanding
  → Best experience

Level 2: No VLM, OCR available
  → Text in images extractable, but no chart understanding
  → Formula images via Nougat (still gets LaTeX)
  → Table images via Docling TableFormer
  → Medium experience

Level 3: No VLM, no OCR
  → Only caption and alt-text preserved
  → Formulas still via Nougat (image-based, no text layer needed)
  → Courseware PDF experience significantly degraded
  → Research paper experience acceptable (text-heavy)
```

System detects document type and warns when VLM is unavailable for courseware.

## 7. Frontend Design

### 7.1 Layout

```
┌──────────────────────────────────────────────────────────────┐
│  ScholarLens   [Document ▼]  [Mode: Parallel | Chat | Notes] [⚙]│
├──────────────────────────────┬───────────────────────────────┤
│                              │                               │
│      Document Reader         │       Interaction Panel       │
│      (PDF.js)                │                               │
│                              │  Chat / Parallel / Notes      │
│                              │  (switchable)                 │
│                              │                               │
│  ┌──────────────────────┐    │  ┌─────────────────────────┐  │
│  │ Section Navigation   │    │  │ Term Panel              │  │
│  │ ▸ 1. Introduction ✓  │    │  │ self-attention 自注意力  │  │
│  │ ▸ 2. Background ✓    │    │  │ positional encoding 位置 │  │
│  │ ▸ 3. Model Arch. →   │    │  └─────────────────────────┘  │
│  └──────────────────────┘    │                               │
├──────────────────────────────┴───────────────────────────────┤
│ 📊 Progress: 3/8 ch | Comprehension: 72% | Tokens: 1,840    │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Three Modes

**Mode 1: Parallel Reading** — English original + Chinese translation side by side (NiceGUI `ui.splitter`), scroll-synced, terms highlighted, hover for details

**Mode 2: Chat Tutoring** — Socratic dialogue with the tutor agent, citation links to document, quick action buttons (continue reading / review / test)

**Mode 3: Learning Notes** — Reading progress overview, term glossary with understanding status, concept relationship graph (Mermaid rendered), export to Obsidian/PDF

### 7.3 Document Reader Interactions

| Interaction | Trigger | Behavior |
|------------|---------|----------|
| **Select text** | Mouse select | Float: [Translate] [Explain] [Ask] |
| **Click term** | Click underlined term | Sidebar shows term detail + related concepts |
| **Page change** | Scroll/click | Update reading position → Tier1 memory update |
| **Click reference** | Click "Figure 3"/"Eq.5" | Jump to figure/formula with explanation |
| **Section nav** | Click navigation tree | Jump to section, pre-load explanation |
| **Highlight** | Select text → highlight | Save to Obsidian notes (bidirectional sync) |

### 7.4 Model Configuration Panel

```
┌─────────────────────────────────────────┐
│ LLM Configuration (required)            │
│ API Key: [________]                      │
│ Base URL: [________]                     │
│ Model: [________]                        │
│                                          │
│ Embedding Model (required)               │
│ API Key: [________]                      │
│ Model: [________]                        │
│                                          │
│ Reranker Model (optional)                │
│ ☐ Use dedicated reranker                 │
│ Model: [________]                        │
│                                          │
│ Vision Model (optional)                  │
│ ☐ Enable vision understanding            │
│ Model: [________]                        │
│                                          │
│ [Test Connection] [Save]                 │
└─────────────────────────────────────────┘
```

## 8. Error Handling & Degradation

### 8.1 Degradation Chain

Every capability follows: **best experience → usable experience → minimum viable**

| Component | Level 1 (Best) | Level 2 (Degraded) | Level 3 (Minimum) |
|-----------|---------------|-------------------|-------------------|
| LLM | Configured model | Cheaper backup model | Rule-based fallback |
| VLM | Full image understanding | OCR text extraction | Caption only |
| Reranker | bge-reranker-m3 | Rule-based reranking | No reranking (RRF only) |
| Vector search | BGE-M3 hybrid | BM25 only | Section navigation matching |
| Parser | Docling + Nougat | PyMuPDF4LLM (no formulas) | pdfplumber basic |
| Memory | Full four-tier | Current session only | No memory |

### 8.2 Circuit Breaker

```python
class CircuitBreaker:
    """3 consecutive failures → open circuit → 60s cooldown → auto-retry"""
    # Applied to: VLM, Reranker, LLM (per-model), Nougat
```

### 8.3 Per-Agent Failure Handling

- **Document Analyzer**: LLM failure → rule-based extraction (regex headings, TF-IDF terms, first-sentence summaries)
- **Content Explainer**: LLM failure → cached term lookup → return original text with apology
- **Validator Agent**: Failure → mark as "unverified", never blocks main flow
- **Learning Tutor**: LLM failure → provide current reading context and suggest retry

## 9. Evaluation System

### 9.1 Six Dimensions

| Dimension | What | How (public datasets only for MVP) |
|-----------|------|-----------------------------------|
| **Document understanding** | Structure/concept extraction accuracy | SciERC entity/relation F1; manual inspection |
| **Translation/explanation** | Translation accuracy, term consistency | COMET/Kiwi automated scoring; term consistency rate |
| **Retrieval quality** | Recall, precision, cross-lingual effectiveness | RAGAS framework (Faithfulness, Context Precision/Recall, Answer Relevancy) |
| **Educational effect** | Actual comprehension improvement | QuALITY long-document QA benchmark; deferred to user study when community grows |
| **Student modeling** | Knowledge tracing prediction accuracy | AUC on interaction logs (after data accumulation) |
| **Token efficiency** | Per-interaction token consumption | Automated tracking via token_tracker.py |

### 9.2 Public Benchmark Datasets

| Dataset | Source | Scale | Use |
|---------|--------|-------|-----|
| **QuALITY** | Public | 200+ long-document QA | Long document understanding baseline |
| **SciERC** | Public | 2,687 paper abstracts | Concept/relation extraction baseline |
| **RAGAS** | Framework | Generate from any QA pairs | RAG quality assessment |
| **COMET-22** | Model | Any translation pairs | Translation quality scoring |

### 9.3 Strategy Comparison Framework

```python
experiments = {
    "baseline": {
        "chunking": "fixed_512",
        "retrieval": "vector_only",
        "reranker": None,
        "memory": "none",
        "context_loading": "flat",
    },
    "v1_optimized": {
        "chunking": "section_aware",
        "retrieval": "hybrid_bm25_vector",
        "reranker": "bge-reranker-m3",
        "memory": "four_tier",
        "context_loading": "L0_L1_L2",
    },
    "v2_full": {
        "chunking": "section_aware_formula_aware",
        "retrieval": "hybrid + contextual_retrieval + CRAG",
        "reranker": "bge-reranker-m3",
        "memory": "four_tier + mermaid_canvas + reflection",
        "context_loading": "L0_L1_L2",
    },
}
```

Automated comparison report generation after each experiment run.

## 10. Project Structure

```
docling/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── agents/
│   │   ├── doc_analyzer.py
│   │   ├── explainer.py
│   │   ├── validator.py
│   │   ├── tutor.py
│   │   └── orchestrator.py
│   ├── memory/
│   │   ├── core_memory.py
│   │   ├── structured_memory.py
│   │   ├── document_memory.py
│   │   ├── reflection_memory.py
│   │   └── memory_manager.py
│   ├── parsers/
│   │   ├── pdf_parser.py
│   │   ├── ppt_parser.py
│   │   ├── chunker.py
│   │   └── reference_extractor.py
│   ├── rag/
│   │   ├── vectorstore.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── contextual_retrieval.py
│   │   └── layered_loader.py
│   ├── core/
│   │   ├── llm_factory.py
│   │   ├── vision_capability.py
│   │   ├── circuit_breaker.py
│   │   ├── models.py
│   │   ├── prompts.py
│   │   ├── settings.py
│   │   ├── exceptions.py
│   │   └── token_tracker.py
│   ├── api/
│   │   ├── middleware.py
│   │   ├── routes/
│   │   │   ├── config.py
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   ├── notes.py
│   │   │   └── memory.py
│   │   └── schemas.py
│   └── knowledge/        # Obsidian output (runtime, gitignored)
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── pdf_viewer.py
│   │   ├── chat_panel.py
│   │   ├── parallel_reader.py
│   │   ├── notes_panel.py
│   │   ├── term_panel.py
│   │   ├── nav_tree.py
│   │   ├── config_panel.py
│   │   └── status_bar.py
│   └── static/
├── data/                  # Runtime data (gitignored)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│       ├── rag_eval.py
│       ├── retrieval_eval.py
│       └── token_eval.py
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 11. Implementation Milestones

```
Week 1-2: Parsing pipeline + chunking + retrieval pipeline
  ├─ Docling integration (PDF + PPTX)
  ├─ Section-aware chunking + reference separation + formula-aware
  ├─ PDF subtype detection
  ├─ BM25 + BGE-M3 + ChromaDB
  ├─ L0/L1/L2 layered summary generation
  └─ Contextual Retrieval prefix generation

Week 3-4: Memory system + 4 agents + LangGraph orchestration
  ├─ Four-tier memory system (SQLite + Markdown)
  ├─ Document Analyzer Agent (structure + concepts + Mermaid + L0/L1/L2)
  ├─ Content Explainer Agent (translation + terms + explanation)
  ├─ Validator Agent (rule + LLM validation)
  ├─ Learning Tutor Agent (Socratic + scaffolding + knowledge tracing)
  └─ LangGraph orchestration (pipeline + interactive + streaming)

Week 5-6: Frontend + integration + degradation + VLM/reranker
  ├─ NiceGUI layout + PDF.js integration
  ├─ Parallel reading + chat tutoring + notes panels
  ├─ VLM integration + degradation strategy
  ├─ bge-reranker-m3 + rule fallback
  ├─ Circuit breaker + all degradation paths
  └─ Token tracking + status bar

Week 7-8: Testing + evaluation + Obsidian output + polish
  ├─ Unit tests + integration tests
  ├─ RAGAS benchmark evaluation
  ├─ Obsidian knowledge base output
  ├─ Concept map visualization (Mermaid)
  ├─ Configuration UX (presets, validation)
  └─ Documentation + open-source readiness
```

## 12. Key Research Influences

| Source | Key Borrowing |
|--------|--------------|
| **OpenViking** (Volcengine) | L0/L1/L2 layered context loading, viking:// filesystem paradigm, directory-recursive retrieval, session self-iteration |
| **TencentDB Agent Memory** | Four-layer semantic pyramid (L0-L3), Mermaid Canvas for token compression, deterministic drill-down traceability, dual storage strategy (DB + Markdown) |
| **OpenClaw** | Workspace-as-context pattern (AGENTS.md/SOUL.md), A2UI live canvas, multi-agent isolation |
| **Hermes Agent** (Nous Research) | Closed-loop learning (skill creation → self-improvement), FTS5 session search, dialectic user modeling, sub-agent RPC delegation |
| **OpenHuman** | Memory Tree + Obsidian Wiki, TokenJuice compression, auto-fetch sync, cross-agent memory interoperability |
| **MemGPT/Letta** | Three-tier memory (Core/Archival/Recall), agent-controlled memory management |
| **ReadAgent** (Google) | Gloss-then-lookup reading pattern, page-level summaries as compressed memory |
| **PaperQA2** (FutureHouse) | Iterative retrieval agent, citation-grounded answers |
| **RAPTOR** | Recursive summary tree for hierarchical document understanding |
| **CRAG** | Corrective RAG — relevance check after retrieval, fallback to LLM knowledge |
| **Contextual Retrieval** (Anthropic) | LLM-generated context prefix per chunk, reduces retrieval failures ~50% |
| **SciERC** | Entity/relation taxonomy for scientific concepts (Used-for, Hyponym-of, Part-of) |
