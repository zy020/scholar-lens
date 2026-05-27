# ScholarLens

ScholarLens is a local-first academic reading assistant for Chinese students reading English research papers and lecture materials. It keeps the source document visible, answers with cited evidence, supports translation and study briefs, and enhances low-quality parses with OCR and optional Vision support.

[中文说明](README_CN.md)

## Highlights

- Evidence-first QA: answers are grounded in retrieved snippets and cite evidence indexes such as `[1]`.
- Paper and courseware workflows: separate upload paths for research papers and courseware; courseware accepts PDF and PPTX.
- Bilingual learning support: Chinese explanations preserve key English terms, formulas, model names, and dataset names.
- Parse-quality diagnostics: heuristic quality scoring recommends OCR or Vision enhancement only where needed.
- OCR/Vision enhancement: GPU RapidOCR runs automatically for recommended pages when CUDA OCR is available; otherwise the UI prompts the user to use Vision or configure GPU OCR.
- Formula-aware text layer: formulas and formula-like text are normalized for retrieval and explanation.
- Semantic chunking and hybrid retrieval: section/page-aware chunks, BM25/token overlap, optional vector search, reranking, context expansion, and memory-aware retrieval hints.
- Study tools: streaming chat, section translation, study briefs, and evidence cards, with learning memory kept as a backend personalization component.

## Roadmap

- Improve PPTX understanding with full-slide rendering before OCR/Vision, especially for screenshot-like slides, dense diagrams, and mixed visual layouts.
- Add stronger table and formula handling beyond the current formula-aware text layer, so numeric table QA and formula explanation become more reliable.
- Move long-running parsing, OCR, Vision, and indexing work to background jobs with progress reporting for larger files.
- Continue improving cross-lingual retrieval quality for Chinese or mixed Chinese-English questions over English papers.

## Architecture

```text
scholar-lens/
├── scholar_lens/
│   ├── api/          FastAPI app, routes, request/response schemas, chat service, and document analysis API
│   │   └── routes/   document upload/parsing/enhancement, config, chat, study brief, and memory APIs
│   ├── parsers/      PDF/PPTX parsing, quality assessment, semantic chunking, OCR/Vision enhancement, and formula text normalization
│   ├── rag/          local document store, BM25/token-overlap retrieval, vector indexing, reranking, and context expansion
│   ├── memory/       learning memory, concept state, current position, recent actions, and retrieval hints
│   ├── agents/       LLM agent wrappers for document understanding, explanation, tutoring, and validation
│   └── core/         settings, model factories, exceptions, circuit breaker, and token tracking infrastructure
├── web/
│   └── src/          React/Vite frontend: sidebar, reader, chat, translate, study brief, and config panels
├── scripts/          local diagnostics, real-file smoke checks, memory smoke checks, and evaluation helpers
├── tests/            backend unit/integration tests and frontend utility tests
└── data/             local runtime data for uploads, parsed documents, chunks, indexes, memory, and eval outputs
```

## Quick Start

```bash
conda create -n scholar_lens python=3.11 -y
conda activate scholar_lens

pip install -e ".[rag,parsers,dev]"

cd web
npm install
cd ..

cp .env.example .env
```

Configure model credentials in `.env` or through the web Config panel. Features that require an LLM show a clear unavailable state until a model is configured.

Start the backend:

```bash
python -m uvicorn scholar_lens.api.main:create_app --factory --reload
```

Start the frontend:

```bash
cd web
npm run dev
```

Vite prints the local frontend URL, typically:

```text
http://localhost:5173
```

## Demo Flow

Use a research paper PDF or lecture PDF/PPTX.

1. Upload a paper through the paper entry, or upload courseware through the courseware entry.
2. Wait until the document status becomes `ready`.
3. GPU OCR enhancement runs automatically for pages recommended by quality diagnostics when available. If GPU OCR is unavailable, the UI prompts for Vision or GPU OCR configuration. Vision and LLM quality review can be enabled in Config when stronger enhancement is needed.
4. Open Reader and select a section.
5. Ask a question in Chat, for example: `这个 attention 公式里的 Q、K、V 分别表示什么？`
6. Watch the Chinese answer stream and expand evidence to verify citations.
7. Use Translate for selected sections or pasted text.
8. Open Study Brief for a structured learning summary.
9. Continue asking follow-up questions; backend learning memory keeps continuity and personalization without appearing as a separate workspace panel.

## Evaluation

ScholarLens was evaluated locally on a self-built 30-question high-confidence QA set covering five documents: three papers (Transformer, BERT, and a GNN survey) and two lecture files (one PDF and one PPTX). The questions cover factual lookup, method understanding, comparison, formula explanation, cross-lingual queries, and lecture-specific structure or figure understanding.

| Metric | Result |
| --- | ---: |
| Generation success rate | 100% |
| Judge success rate | 100% |
| Citation rate | 96.7% |
| Empty context rate | 0.0% |
| Correctness | 3.37 / 5 |
| Faithfulness | 4.10 / 5 |
| Evidence quality | 3.67 / 5 |
| Completeness | 3.03 / 5 |
| Retrieval hit@5 | 16.7% |
| Context hit@5 | 16.7% |
| MRR@5 | 0.15 |

## License

MIT
