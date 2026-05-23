# Doc-Type & Text-Quality Aware Brief Feedback

## Automatic Verification

```
pytest command:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/unit/test_pdf_diagnostics.py tests/unit/api/test_documents.py tests/unit/api/test_notes_brief.py -q
pytest result:   26 passed, 1 warning (pre-existing asyncio_mode)
npm test:        13 passed, 0 failed
npm run lint:    exit 0
npm run build:   exit 0
compileall:      exit 0
```

## Manual Browser Verification

| Case | File Used | Expected | Screenshot | Result |
|------|-----------|----------|------------|--------|
| Paper PDF | transformer_attention.pdf | Paper Brief with good text | docs/screenshots/doc-type-paper-brief.png | NOT CAPTURED (backend restart blocked) |
| Text lecture PDF | — | Lecture Brief | docs/screenshots/doc-type-lecture-brief.png | NOT CAPTURED (need text lecture PDF) |
| Image-based/weak lecture PDF | MIT image slides | Low-text warning | docs/screenshots/doc-type-low-text-warning.png | NOT CAPTURED (backend restart blocked) |

## Code Evidence

- Document diagnostic fields: DocumentSummary.text_quality, ocr_needed, page_text_coverage, section_quality, diagnostic_notes
- Parser diagnostics: diagnose_text_quality() in pdf_parser.py — 3 unit tests
- Brief routing: _is_lecture_doc(), _needs_low_text_brief() in notes.py — 2 routing tests
- Frontend display: briefTitle() switch, type-specific section headings, text quality badge

## Known Limits

- OCR is not implemented in this phase.
- Vision model path is not implemented in this phase.
- Manual screenshots not captured due to backend restart blocked by auto mode.
- Lecture screenshots require a text-selectable lecture PDF (none currently in test_docs).
