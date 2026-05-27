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

Respond with a JSON object containing:
- doc_type: "research_paper" | "courseware" | "textbook_chapter"
- language: "en" | "zh" | "mixed"
- difficulty: "beginner" | "intermediate" | "advanced"
- estimated_reading_time: integer (minutes)
- sections: list of {{"section_id": string, "title": string, "level": integer, "page_start": integer or null, "page_end": integer or null, "section_type": "prose"|"method"|"results"}}
- key_terms: list of {{"english": string, "chinese": string, "relation_type": "Used-for"|"Hyponym-of"|"Part-of"|"Compare-with"|null}}. Infer relation types between terms when possible (e.g. "multi-head attention" Hyponym-of "attention", "softmax" Used-for "attention calculation")
- prerequisites: list of prerequisite concept names
- mermaid_map: Mermaid mindmap or graph showing the document structure (use mindmap or graph TD syntax, include all sections)
"""

DOC_ANALYZER_L0 = """Generate a very short L0 summary (~100 tokens) for the following document section.
This summary will be used for quick retrieval — it should capture only the most essential point.

Section title: {title}
Section text (first 2000 chars):
{text}

L0 summary (1-2 sentences, ~100 tokens):"""

DOC_ANALYZER_L1 = """Generate a detailed L1 overview (~2000 tokens) for the following document section.
Include key concepts, methodology, findings, and important technical details.

Section title: {title}
Section text (first 5000 chars):
{text}

L1 overview (~2000 tokens):"""

DOC_ANALYZER_TERMS = """Extract 5-10 key technical terms from the following document excerpt.
For each term, provide the English term and its Chinese translation.
Focus on domain-specific terms that a Chinese student would need to understand.

Document excerpt:
{text}

Respond with a JSON list: [{{"english": "...", "chinese": "..."}}, ...]"""

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
