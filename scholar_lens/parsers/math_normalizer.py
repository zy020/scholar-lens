from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


MATH_SYMBOL_ALIASES = {
    "𝛼": "alpha",
    "α": "alpha",
    "𝛽": "beta",
    "β": "beta",
    "𝛾": "gamma",
    "γ": "gamma",
    "𝜆": "lambda",
    "λ": "lambda",
    "𝜇": "mu",
    "μ": "mu",
    "𝜎": "sigma",
    "σ": "sigma",
    "𝜃": "theta",
    "θ": "theta",
    "∙": " dot ",
    "·": " dot ",
    "⋅": " dot ",
    "×": " times ",
    "⊙": " elementwise product ",
    "√": " sqrt ",
    "∑": " sum ",
    "Σ": " sum ",
    "∫": " integral ",
    "∞": " infinity ",
    "≤": " <= ",
    "≥": " >= ",
    "≠": " != ",
    "≈": " approx ",
    "′": " prime ",
    "’": " prime ",
}

LATEX_ALIASES = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\theta": "theta",
    r"\cdot": " dot ",
    r"\times": " times ",
    r"\sqrt": " sqrt ",
    r"\sum": " sum ",
    r"\int": " integral ",
    r"\infty": " infinity ",
    r"\leq": " <= ",
    r"\geq": " >= ",
    r"\neq": " != ",
    r"\approx": " approx ",
}

MATH_KEYWORDS = {
    "alpha", "beta", "gamma", "lambda", "mu", "sigma", "theta", "dot", "times",
    "sqrt", "sum", "integral", "softmax", "argmax", "log", "loss", "matrix",
}
MATH_LIKE_RE = re.compile(
    r"[\$\\=+\-*/^_{}()[\]<>≤≥≠≈∙·⋅×⊙√∑Σ∫∞𝒂-𝒛𝑨-𝒁𝑎-𝑧𝛼-𝜔α-ω]"
)


@dataclass(frozen=True)
class MathTextAnalysis:
    has_formula: bool
    normalized_text: str = ""
    symbols: list[str] | None = None
    formula_ids: list[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", self.symbols or [])
        object.__setattr__(self, "formula_ids", self.formula_ids or [])


def normalize_math_text(text: str) -> str:
    normalized = text or ""
    for latex, alias in LATEX_ALIASES.items():
        normalized = normalized.replace(latex, alias)
    chars = []
    for ch in normalized:
        chars.append(MATH_SYMBOL_ALIASES.get(ch, _normalize_char(ch)))
    normalized = "".join(chars)
    normalized = normalized.replace("^T", " transpose ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", normalized)
    normalized = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def analyze_math_text(text: str) -> MathTextAnalysis:
    if not text:
        return MathTextAnalysis(has_formula=False)

    normalized = normalize_math_text(text)
    symbols = _extract_symbols(normalized)
    has_formula = _looks_formula_like(text, normalized, symbols)
    if not has_formula:
        return MathTextAnalysis(has_formula=False)

    formula_ids = _formula_terms(normalized, symbols)
    return MathTextAnalysis(
        has_formula=True,
        normalized_text=normalized,
        symbols=symbols,
        formula_ids=formula_ids,
    )


def _normalize_char(ch: str) -> str:
    try:
        normalized = unicodedata.normalize("NFKC", ch)
    except Exception:
        return ch
    return normalized if normalized else ch


def _extract_symbols(normalized: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]+|[0-9]+", normalized)
    symbols = []
    for token in tokens:
        lowered = token.lower()
        if len(lowered) == 1 or lowered in MATH_KEYWORDS:
            symbols.append(lowered)
    return _unique(symbols)


def _looks_formula_like(text: str, normalized: str, symbols: list[str]) -> bool:
    if "$" in text or "\\" in text:
        return True
    if any(alias in normalized.lower().split() for alias in MATH_KEYWORDS):
        return True
    if MATH_LIKE_RE.search(text) and len(symbols) >= 2:
        return True
    if any(ch in text for ch in "=+*/^_≤≥≠≈∙·⋅×⊙√∑Σ∫"):
        return True
    if "-" in text and len(symbols) >= 2 and re.search(r"\b[A-Za-z]\s*-\s*[A-Za-z]\b", text):
        return True
    return False


def _formula_terms(normalized: str, symbols: list[str]) -> list[str]:
    terms = []
    lowered = normalized.lower()
    for pattern in (r"\bq\s+dot\s+k\b", r"\bk\s+transpose\b", r"\bsoftmax\b", r"\balpha\b"):
        match = re.search(pattern, lowered)
        if match:
            terms.append(match.group(0))
    if symbols:
        terms.append(" ".join(symbols[:12]))
    if lowered:
        terms.append(lowered[:160])
    return _unique(terms)


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = " ".join(str(item).split())
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
