from scholar_lens.core.token_tracker import estimate_tokens
from scholar_lens.parsers.chunker import DEFAULT_CHUNKING_POLICIES, SectionAwareChunker
from scholar_lens.parsers.models import ParsedDocument, ParsedPage


class TestSectionAwareChunker:
    def _make_doc(self, sections=None, raw_text=""):
        return ParsedDocument(source_path="test.pdf", doc_subtype="research_paper", sections=sections or [], raw_text=raw_text)

    def test_chunk_by_section(self):
        doc = self._make_doc(
            sections=[
                {"id": "1", "title": "Intro", "level": 1, "text": "A" * 500},
                {"id": "2", "title": "Method", "level": 1, "text": "B" * 500},
            ],
            raw_text="A" * 500 + "\n" + "B" * 500,
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="test_doc")
        assert len(chunks) >= 2
        assert chunks[0].metadata.section_id == "1"
        assert chunks[1].metadata.section_id == "2"

    def test_sections_without_text_fall_back_to_raw_text(self):
        doc = self._make_doc(
            sections=[
                {"id": "1", "title": "Introduction", "level": 1, "page_start": 1},
                {"id": "2", "title": "Method", "level": 1, "page_start": 2},
            ],
            raw_text="Attention Is All You Need\n\nThe encoder is composed of a stack of N = 6 identical layers.",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=80, overlap_tokens=8)

        chunks = chunker.chunk(doc, doc_id="paper")

        assert len(chunks) == 1
        assert "N = 6 identical layers" in chunks[0].text
        assert chunks[0].metadata.section_id == "0"

    def test_large_section_split(self):
        doc = self._make_doc(sections=[{"id": "1", "title": "Long", "level": 1, "text": "X" * 3000}], raw_text="X" * 3000)
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="test_doc")
        assert len(chunks) > 1

    def test_default_policy_uses_document_type(self):
        paper = self._make_doc(sections=[{"id": "1", "title": "Intro", "text": "Paper text."}], raw_text="Paper text.")
        slides = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[ParsedPage(page_num=0, text="Slide text.", char_count=11)],
            raw_text="Slide text.",
        )
        chunker = SectionAwareChunker()

        chunker.chunk(paper, doc_id="paper")
        assert chunker.max_chunk_tokens == DEFAULT_CHUNKING_POLICIES["research_paper"].max_chunk_tokens
        assert chunker.overlap_tokens == DEFAULT_CHUNKING_POLICIES["research_paper"].overlap_tokens

        chunker.chunk(slides, doc_id="slides")
        assert chunker.max_chunk_tokens == DEFAULT_CHUNKING_POLICIES["slides_pdf"].max_chunk_tokens
        assert chunker.overlap_tokens == DEFAULT_CHUNKING_POLICIES["slides_pdf"].overlap_tokens

    def test_explicit_policy_overrides_document_type_defaults(self):
        slides = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[ParsedPage(page_num=0, text="Slide text.", char_count=11)],
            raw_text="Slide text.",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=900, overlap_tokens=120)

        chunker.chunk(slides, doc_id="slides")

        assert chunker.max_chunk_tokens == 900
        assert chunker.overlap_tokens == 120

    def test_large_section_split_uses_overlap(self):
        paragraphs = [
            " ".join(f"alpha{i}" for i in range(60)),
            " ".join(f"bridge{i}" for i in range(60)),
            " ".join(f"omega{i}" for i in range(60)),
        ]
        doc = self._make_doc(
            sections=[{"id": "1", "title": "Long", "level": 1, "text": "\n\n".join(paragraphs)}],
            raw_text="\n\n".join(paragraphs),
        )
        chunker = SectionAwareChunker(max_chunk_tokens=80, overlap_tokens=8)

        chunks = chunker.chunk(doc, doc_id="test_doc")

        assert len(chunks) > 1
        tail = " ".join(chunks[0].text.split()[-8:])
        assert chunks[1].text.startswith(tail)

    def test_overlap_is_trimmed_to_respect_chunk_limit(self):
        paragraphs = [
            " ".join(f"alpha{i}" for i in range(45)),
            " ".join(f"beta{i}" for i in range(45)),
        ]
        doc = self._make_doc(
            sections=[{"id": "1", "title": "Long", "level": 1, "text": "\n\n".join(paragraphs)}],
            raw_text="\n\n".join(paragraphs),
        )
        chunker = SectionAwareChunker(max_chunk_tokens=80, overlap_tokens=30)

        chunks = chunker.chunk(doc, doc_id="test_doc")

        assert len(chunks) > 1
        assert all(estimate_tokens(chunk.text) <= 80 for chunk in chunks)

    def test_oversized_paragraph_split_uses_overlap(self):
        text = " ".join(f"word{i}" for i in range(180))
        doc = self._make_doc(sections=[{"id": "1", "title": "Long", "level": 1, "text": text}], raw_text=text)
        chunker = SectionAwareChunker(max_chunk_tokens=40, overlap_tokens=6)

        chunks = chunker.chunk(doc, doc_id="test_doc")

        assert len(chunks) > 1
        tail = " ".join(chunks[0].text.split()[-6:])
        assert chunks[1].text.startswith(tail)

    def test_semantic_heading_stays_with_following_body(self):
        intro = " ".join(f"intro{i}" for i in range(40))
        method_body = "We train the model with contrastive batches and evaluate retrieval quality."
        text = f"{intro}\n\n## Method\n\n{method_body}"
        doc = self._make_doc(sections=[{"id": "1", "title": "Paper", "level": 1, "text": text}], raw_text=text)
        chunker = SectionAwareChunker(max_chunk_tokens=90, overlap_tokens=8)

        chunks = chunker.chunk(doc, doc_id="paper")

        method_chunk = next(chunk for chunk in chunks if "## Method" in chunk.text)
        assert method_body in method_chunk.text
        assert not chunks[0].text.rstrip().endswith("## Method")

    def test_semantic_bullet_list_stays_together_when_it_fits(self):
        text = (
            "Training recipe:\n"
            "- sample hard negatives from the same paper\n"
            "- cache embeddings for repeated retrieval\n"
            "- rerank the top passages with a cross encoder\n\n"
            "The recipe improves answer grounding."
        )
        doc = self._make_doc(sections=[{"id": "1", "title": "Method", "level": 1, "text": text}], raw_text=text)
        chunker = SectionAwareChunker(max_chunk_tokens=80, overlap_tokens=6)

        chunks = chunker.chunk(doc, doc_id="paper")

        list_chunk = next(chunk for chunk in chunks if "- sample hard negatives" in chunk.text)
        assert "- cache embeddings for repeated retrieval" in list_chunk.text
        assert "- rerank the top passages with a cross encoder" in list_chunk.text

    def test_semantic_formula_and_caption_blocks_stay_intact_when_they_fit(self):
        text = (
            "The loss combines alignment and uniformity.\n\n"
            "$$\n"
            "L = L_align + lambda L_uniform\n"
            "$$\n\n"
            "Figure 2: Retrieval quality improves after semantic chunking.\n\n"
            "The caption explains the trend in the plot."
        )
        doc = self._make_doc(sections=[{"id": "1", "title": "Results", "level": 1, "text": text}], raw_text=text)
        chunker = SectionAwareChunker(max_chunk_tokens=40, overlap_tokens=6)

        chunks = chunker.chunk(doc, doc_id="paper")

        formula_chunk = next(chunk for chunk in chunks if "L_align" in chunk.text)
        assert "$$" in formula_chunk.text
        caption_chunk = next(chunk for chunk in chunks if "Figure 2:" in chunk.text)
        assert "The caption explains the trend in the plot." in caption_chunk.text

    def test_paper_caption_does_not_swallow_following_numbered_heading(self):
        text = (
            "Figure 3: Retrieval accuracy by question type.\n\n"
            "4 Results\n\n"
            "The result section compares retrieval and reranking quality."
        )
        doc = self._make_doc(sections=[{"id": "3", "title": "Figures", "level": 1, "text": text}], raw_text=text)
        chunker = SectionAwareChunker(max_chunk_tokens=30, overlap_tokens=4)

        chunks = chunker.chunk(doc, doc_id="paper")

        caption_chunk = next(chunk for chunk in chunks if "Figure 3:" in chunk.text)
        results_chunk = next(chunk for chunk in chunks if "4 Results" in chunk.text)
        assert "4 Results" not in caption_chunk.text
        assert "The result section compares retrieval" in results_chunk.text

    def test_courseware_slide_title_stays_with_following_bullet_group(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[
                ParsedPage(
                    page_num=0,
                    text=(
                        "Optimization Tricks\n\n"
                        "- warm up the learning rate\n"
                        "- clip gradients\n"
                        "- monitor validation loss"
                    ),
                    char_count=92,
                )
            ],
            raw_text="Optimization Tricks",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=25, overlap_tokens=4)

        chunks = chunker.chunk(doc, doc_id="slides")

        assert len(chunks) == 1
        assert chunks[0].text.startswith("Optimization Tricks")
        assert "- monitor validation loss" in chunks[0].text

    def test_courseware_teaching_label_stays_with_following_explanation(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[
                ParsedPage(
                    page_num=0,
                    text=(
                        "Attention Basics\n\n"
                        "Definition:\n\n"
                        "Attention maps a query to weighted values using compatibility scores."
                    ),
                    char_count=96,
                )
            ],
            raw_text="Attention Basics",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=22, overlap_tokens=4)

        chunks = chunker.chunk(doc, doc_id="slides")

        definition_chunk = next(chunk for chunk in chunks if "Definition:" in chunk.text)
        assert "Attention maps a query" in definition_chunk.text
        assert not chunks[0].text.rstrip().endswith("Definition:")

    def test_chunk_metadata_populated(self):
        doc = self._make_doc(sections=[{"id": "3.1", "title": "Model", "level": 2, "text": "Short text about the model."}], raw_text="Short text about the model.")
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="paper_001")
        assert len(chunks) == 1
        assert chunks[0].metadata.section_id == "3.1"
        assert chunks[0].metadata.doc_id == "paper_001"
        assert chunks[0].layer == "L2"

    def test_empty_document(self):
        doc = self._make_doc(raw_text="")
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="empty")
        assert chunks == []

    def test_chunk_id_format(self):
        doc = self._make_doc(sections=[{"id": "1", "title": "A", "level": 1, "text": "Hello world"}], raw_text="Hello world")
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="p001")
        assert chunks[0].chunk_id.startswith("p001_1_")

    def test_references_section_skipped(self):
        doc = self._make_doc(
            sections=[
                {"id": "1", "title": "Intro", "level": 1, "text": "Some intro text"},
                {"id": "ref", "title": "References", "level": 1, "text": "[1] Smith. Paper. 2020."},
            ],
            raw_text="Some intro text\n\nReferences\n[1] Smith. Paper. 2020.",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)
        chunks = chunker.chunk(doc, doc_id="test")
        assert len(chunks) == 1
        assert chunks[0].metadata.section_id == "1"

    def test_courseware_pdf_chunks_preserve_slide_sections(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            sections=[
                {"id": "slide_0", "title": "Attention", "text": "Query key value overview."},
                {"id": "slide_1", "title": "Training", "text": "Loss curves and examples."},
            ],
            raw_text="Attention\n\nTraining",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)

        chunks = chunker.chunk(doc, doc_id="deck")

        assert [c.metadata.section_id for c in chunks] == ["slide_0", "slide_1"]
        assert all(c.metadata.content_type == "slide" for c in chunks)
        assert chunks[0].text.startswith("Attention")
        assert chunks[0].chunk_id == "deck_slide_0_0"

    def test_courseware_pdf_uses_pages_for_untitled_slides(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[
                ParsedPage(page_num=0, text="Intro\nCourse goals.", char_count=19),
                ParsedPage(page_num=1, text="Q K V attention details.", char_count=22),
            ],
            sections=[
                {"id": "slide_0", "title": "Intro", "text": "Intro\nCourse goals."},
            ],
            raw_text="Intro\n\nQ K V attention details.",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)

        chunks = chunker.chunk(doc, doc_id="deck")

        assert [c.metadata.section_id for c in chunks] == ["slide_0", "slide_1"]
        assert chunks[1].text.startswith("Q K V attention details.")
        assert chunks[1].metadata.page_start == 1
        assert chunks[1].metadata.page_end == 1

    def test_slides_pdf_chunks_pages_when_sections_missing(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[
                ParsedPage(page_num=0, text="Agenda\nToday we study transformers.", char_count=35),
                ParsedPage(page_num=1, text="Self Attention\nCompute weighted token context.", char_count=45),
            ],
            raw_text="Agenda\n\nSelf Attention",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)

        chunks = chunker.chunk(doc, doc_id="slides")

        assert [c.metadata.section_id for c in chunks] == ["slide_0", "slide_1"]
        assert all(c.metadata.content_type == "slide" for c in chunks)
        assert chunks[1].metadata.chapter == "2"
        assert chunks[0].metadata.page_start == 0
        assert chunks[0].metadata.page_end == 0

    def test_courseware_chunks_preserve_enhancement_source(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[
                ParsedPage(
                    page_num=3,
                    text="Recognized OCR slide text.",
                    char_count=26,
                    content_source="ocr",
                    enhanced=True,
                ),
            ],
            raw_text="Recognized OCR slide text.",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)

        chunks = chunker.chunk(doc, doc_id="slides")

        assert chunks[0].metadata.page_start == 3
        assert chunks[0].metadata.page_end == 3
        assert chunks[0].metadata.content_source == "ocr"
        assert chunks[0].metadata.enhanced is True

    def test_chunk_metadata_includes_normalized_formula_terms(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[
                ParsedPage(
                    page_num=4,
                    text="Dot-product attention\n𝛼 = softmax(𝒒∙𝒌)",
                    char_count=43,
                )
            ],
            raw_text="Dot-product attention\n𝛼 = softmax(𝒒∙𝒌)",
        )
        chunker = SectionAwareChunker(max_chunk_tokens=600, overlap_tokens=50)

        chunks = chunker.chunk(doc, doc_id="deck")

        assert chunks[0].text == "Dot-product attention\n𝛼 = softmax(𝒒∙𝒌)"
        assert chunks[0].metadata.has_formula is True
        assert any("q dot k" in item for item in chunks[0].metadata.formula_ids)
        assert "Formula terms:" in chunks[0].metadata.contextual_prefix
        assert "alpha" in chunks[0].metadata.contextual_prefix

    def test_chunk_metadata_marks_vision_structured_formula_and_table_content(self):
        doc = ParsedDocument(
            source_path="slides.pdf",
            doc_subtype="slides_pdf",
            pages=[
                ParsedPage(
                    page_num=3,
                    content_source="vision",
                    enhanced=True,
                    text=(
                        "The slide defines scaled dot-product attention.\n"
                        "Visual type: formula\n"
                        "Formula summary: Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V\n"
                        "QA hint: Useful for formula questions."
                    ),
                ),
                ParsedPage(
                    page_num=4,
                    content_source="vision",
                    enhanced=True,
                    text=(
                        "The table compares model accuracy.\n"
                        "Visual type: table\n"
                        "Table summary: Model A reaches 91% and Model B reaches 88%."
                    ),
                ),
            ],
        )

        chunks = SectionAwareChunker().chunk(doc, doc_id="doc1")

        formula_chunk = next(chunk for chunk in chunks if "scaled dot-product" in chunk.text)
        table_chunk = next(chunk for chunk in chunks if "Model A reaches" in chunk.text)
        assert formula_chunk.metadata.has_formula is True
        assert formula_chunk.metadata.content_type == "formula"
        assert "Vision formula summary" in formula_chunk.metadata.contextual_prefix
        assert table_chunk.metadata.content_type == "table"
        assert table_chunk.metadata.caption == "Model A reaches 91% and Model B reaches 88%."
