# ScholarLens

ScholarLens is a local-first academic reading assistant for Chinese students reading English research papers and lecture materials. It keeps the source document visible, answers with retrieved document context, supports translation and document analysis, and enhances low-quality parses with OCR and optional Vision support.

[中文](README_CN.md)

## Highlights

- Context-grounded QA: answers use retrieved document snippets while still allowing clearly separated educational background when helpful.
- Explicit document type selection: a single PDF upload entry lets users choose paper or courseware before upload.
- Bilingual learning support: Chinese explanations preserve key English terms, formulas, model names, and dataset names.
- Parse-quality diagnostics: heuristic quality scoring recommends OCR or Vision enhancement only where needed.
- Parse-quality enhancement: GPU RapidOCR runs automatically for recommended OCR pages when available; the frontend can enable LLM quality review and Vision in the Config panel and manually run parse enhancement for difficult pages.
- Formula-aware text layer: formula-like text is normalized and tagged to improve retrieval and explanation.
- Semantic chunking and hybrid retrieval: section/page-aware chunks, keyword/BM25 retrieval by default, optional vector search, optional model reranking, context expansion, and memory-aware retrieval hints.
- Study tools: streaming chat, paper section translation, selected-text translation for courseware, document learning analysis, and backend learning memory for personalization.

## Roadmap

- Improve PDF courseware parsing for screenshot-like slides, dense diagrams, tables, formulas, and mixed visual layouts.
- Move long-running parsing, OCR, Vision, and indexing work to background jobs with clearer progress reporting for larger files.
- Continue improving cross-lingual retrieval quality for Chinese or mixed Chinese-English questions over English papers.
- Further improve formula, chart, and table understanding; formulas are currently handled mainly through text normalization, while complex visual semantics still rely on Vision.
- Improve model-call stability, retries, and task recovery so external API instability affects long workflows less.

## Supported Documents

- Research paper PDFs.
- Courseware PDFs.
- PPT, PPTX, Word, and image files are not accepted directly. Export slides or documents to PDF before upload.

Paper and courseware share one upload control, but the user explicitly selects the document type before uploading. This lets ScholarLens apply different parsing, chunking, retrieval, translation, and analysis strategies.

## Architecture

```text
scholar-lens/
├── scholar_lens/
│   ├── api/          FastAPI app, routes, request/response schemas, chat service, and document analysis API
│   │   └── routes/   document upload/parsing/enhancement, config, chat, document analysis, and memory APIs
│   ├── parsers/      PDF parsing, quality assessment, semantic chunking, OCR/Vision enhancement, and formula text normalization
│   ├── rag/          local document store, BM25/token-overlap retrieval, vector indexing, reranking, and context expansion
│   ├── memory/       learning memory, concept state, current position, recent actions, and retrieval hints
│   ├── agents/       LLM agent wrappers for document understanding, explanation, tutoring, and validation
│   └── core/         settings, model factories, exceptions, circuit breaker, and token tracking infrastructure
├── web/
│   └── src/          React/Vite frontend: sidebar, reader, chat, translate, document analysis, and config panels
├── tests/            backend unit/integration tests and frontend utility tests
└── data/             local runtime data for uploads, parsed documents, chunks, indexes, memory, and caches
```

## Requirements

- Python 3.11.
- Node.js and npm for the React/Vite frontend.
- A configured LLM for chat, translation, document learning analysis, and optional LLM parse-quality review.
- An embedding model is recommended for vector retrieval; the system can still fall back to keyword/BM25 retrieval when embeddings are unavailable.
- GPU OCR requires RapidOCR with ONNX Runtime CUDA support. If CUDA OCR is unavailable, OCR is paused; manual parse enhancement can still use LLM quality review and Vision enhancement when those models are configured.

## Configuration

Model settings can be provided through `.env` or updated from the web Config panel. Shared `API_KEY` and `BASE_URL` are inherited by LLM, embedding, reranker, and Vision models unless a model-specific override is configured.

Common settings include:

- `LLM__MODEL` for chat, translation, document analysis, and optional LLM quality review.
- `EMBEDDING__MODEL` for vector indexing and retrieval.
- `RERANKER__MODEL` for optional model-based reranking.
- `VISION__MODEL` for optional Vision-based parse enhancement.

## Data & Privacy

ScholarLens stores uploaded files, parsed documents, chunks, indexes, cached analysis, and memory data locally under `data/` by default. When LLM, embedding, reranker, or Vision features are enabled, the relevant text, query, retrieved context, or selected page images may be sent to the configured model provider.

## Limitations

- Direct PPT/PPTX upload is not supported; export slides to PDF first.
- Screenshot-heavy lecture PDFs, dense visual diagrams, complex tables, and formula-heavy pages may require Vision enhancement for better understanding.
- Formula understanding currently relies mostly on text normalization, LaTeX/symbol preservation, and retrieval augmentation; it is not a full formula OCR or automatic derivation system.
- Chart and table parsing uses lightweight structure detection plus text enhancement; complex visual relations still need Vision model support.
- Large PDFs can take noticeable time to parse, enhance, and index because long-running work is still handled in the request flow.
- The project is currently best suited as a local single-user learning tool; multi-user authorization, job queues, and cloud deployment governance are not implemented.
- The local evaluation set is not included because it contains private courseware examples.

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
http://localhost:3000
```

## Evaluation

ScholarLens was evaluated locally for RAG on a self-built 30-question high-confidence QA set covering five documents: three papers (Transformer, BERT, and a GNN survey) and two lecture PDFs. The questions cover factual lookup, method understanding, comparison, formula explanation, cross-lingual queries, and lecture-specific structure or figure understanding. The dataset is not published because it includes private courseware.

| Metric | Result |
| --- | ---: |
| Generation success rate | 100% |
| Judge success rate | 100% |
| Empty context rate | 0.0% |
| Correctness | 3.37 / 5 |
| Faithfulness | 4.10 / 5 |
| Completeness | 3.03 / 5 |

## License

MIT
