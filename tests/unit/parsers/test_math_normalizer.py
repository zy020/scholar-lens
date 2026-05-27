from scholar_lens.parsers.math_normalizer import analyze_math_text, normalize_math_text


def test_normalize_unicode_attention_symbols_to_search_terms():
    normalized = normalize_math_text("𝛼 = 𝒒∙𝒌, 𝑊𝑞 projects queries")

    assert "alpha" in normalized
    assert "q dot k" in normalized
    assert "Wq" in normalized
    assert "projects queries" in normalized


def test_analyze_math_text_extracts_symbols_and_formula_terms():
    analysis = analyze_math_text("Self-attention uses 𝛼 = softmax(𝒒∙𝒌).")

    assert analysis.has_formula is True
    assert "alpha" in analysis.normalized_text
    assert "q" in analysis.symbols
    assert "k" in analysis.symbols
    assert "alpha" in analysis.symbols
    assert analysis.formula_ids
    assert any("q dot k" in item for item in analysis.formula_ids)


def test_analyze_plain_text_is_not_formula_heavy():
    analysis = analyze_math_text("Self-attention compares token representations.")

    assert analysis.has_formula is False
    assert analysis.normalized_text == ""
    assert analysis.symbols == []
    assert analysis.formula_ids == []
