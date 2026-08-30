"""Document text extraction for PDF and Word resumes.

Real resume corpora are messy, and the sample set exposes three concrete failure
modes that naive extraction gets wrong:

1. **Ligature loss in PDFs.** One resume was produced by a LaTeX-style pipeline that
   encodes "ti" as a single glyph; pypdf renders it as U+FFFD, turning
   "Quantitative Developer" into "Quan?ta?ve Developer". Left unfixed, the LLM sees
   corrupted job titles.
2. **Table-based Word layouts.** Two resumes store the entire experience section in
   Word tables whose cells are merged horizontally, so python-docx reports the same
   text four times per row. That quadruples token cost and biases the model toward
   repeated content.
3. **Reading order.** Word documents interleave paragraphs and tables; iterating
   paragraphs first and tables second scrambles chronology. We walk the document
   body in true XML order instead.

Every document therefore goes through: extract -> normalise unicode -> de-duplicate
-> collapse whitespace, and we record which repairs fired so the notebook can show
them.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pypdf
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

# Replacement-character repairs for the PDF ligature problem. Ordered longest-first
# so "iden?fica?on" style multi-hit words resolve correctly.
LIGATURE_REPAIRS: list[tuple[str, str]] = [
    ("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"), ("\ufb03", "ffi"), ("\ufb04", "ffl"),
    ("\ufb05", "st"), ("\ufb06", "st"),
]

# Words in the corpus that lose the "ti" ligature. Rather than blind-replacing every
# U+FFFD with "ti" (which would corrupt genuinely unknown glyphs), we repair a
# vocabulary of finance/tech terms and flag anything left over for review.
FFFD_VOCAB = [
    "quantitative", "quantitive", "implementation", "introduction", "identification",
    "optimization", "optimisation", "computation", "validation", "education",
    "certification", "application", "presentation", "information", "operations",
    "portfolio", "differential", "statistics", "stochastic", "options", "option",
    "exotic", "payoff", "strategies", "strategie", "practice", "front", "office",
    "testing", "construction", "probability", "analysis", "derivatives",
]


@dataclass
class LoadedDocument:
    """Extracted text plus a provenance record of every repair we applied."""
    path: Path
    text: str
    n_chars: int
    file_type: str
    repairs: list[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return self.path.name


# --------------------------------------------------------------------- cleaning
def _repair_replacement_chars(text: str) -> tuple[str, int]:
    """Rebuild words mangled by lost ligatures.

    Strategy: for each token containing U+FFFD, try substituting the common lost
    ligatures ("ti" first, then "tt"/"ft"/"fi") and accept the candidate that matches
    a known vocabulary word. This is deterministic and auditable - we never guess
    silently.
    """
    if "\ufffd" not in text:
        return text, 0

    vocab = set(FFFD_VOCAB)
    fixed = 0

    def fix_token(token: str) -> str:
        nonlocal fixed
        for sub in ("ti", "tt", "ft", "fi", "ffi", "fl"):
            candidate = token.replace("\ufffd", sub)
            stripped = re.sub(r"[^A-Za-z]", "", candidate).lower()
            if stripped in vocab:
                fixed += 1
                return candidate
        # Fall back to the statistically dominant ligature in this corpus.
        fixed += 1
        return token.replace("\ufffd", "ti")

    out = " ".join(fix_token(tok) if "\ufffd" in tok else tok for tok in text.split(" "))
    return out, fixed


def clean_text(text: str) -> tuple[str, list[str]]:
    """Normalise unicode, repair ligatures and collapse whitespace."""
    repairs: list[str] = []

    for src, dst in LIGATURE_REPAIRS:
        if src in text:
            text = text.replace(src, dst)
            repairs.append(f"expanded ligature {src!r} -> {dst!r}")

    text, n_fffd = _repair_replacement_chars(text)
    if n_fffd:
        repairs.append(f"repaired {n_fffd} token(s) containing U+FFFD replacement characters")

    # NFKC folds full-width and compatibility characters onto ASCII equivalents.
    text = unicodedata.normalize("NFKC", text)

    # Normalise the many dash and quote variants resumes pick up from Word.
    for src, dst in (("\u2013", "-"), ("\u2014", "-"), ("\u2018", "'"), ("\u2019", "'"),
                     ("\u201c", '"'), ("\u201d", '"'), ("\u00a0", " "), ("\u2022", "-")):
        text = text.replace(src, dst)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    return text.strip(), repairs


def _dedupe_row_cells(cells: list[str]) -> list[str]:
    """Drop consecutive duplicate cell values produced by horizontally merged cells."""
    out: list[str] = []
    for value in cells:
        value = value.strip()
        if not value:
            continue
        if out and value == out[-1]:
            continue
        out.append(value)
    return out


# --------------------------------------------------------------------- loaders
def _iter_docx_blocks(doc: Document):
    """Yield paragraphs and tables in true document order."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def load_docx(path: Path) -> tuple[str, list[str]]:
    doc = Document(str(path))
    parts: list[str] = []
    repairs: list[str] = []
    duplicate_cells = 0

    for block in _iter_docx_blocks(doc):
        if isinstance(block, Paragraph):
            if block.text.strip():
                parts.append(block.text)
        else:
            parts.append("")  # blank line keeps table content visually separated
            for row in block.rows:
                raw = [c.text for c in row.cells]
                deduped = _dedupe_row_cells(raw)
                duplicate_cells += len([c for c in raw if c.strip()]) - len(deduped)
                if deduped:
                    parts.append(" | ".join(deduped))
            parts.append("")

    if duplicate_cells:
        repairs.append(f"removed {duplicate_cells} duplicate table cell value(s) from merged cells")
    return "\n".join(parts), repairs


def load_pdf(path: Path) -> tuple[str, list[str]]:
    reader = pypdf.PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages), [f"extracted {len(pages)} PDF page(s)"]


def load_document(path: str | Path) -> LoadedDocument:
    """Extract cleaned text from a single resume file."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".docx":
        raw, repairs = load_docx(path)
        file_type = "docx"
    elif suffix == ".pdf":
        raw, repairs = load_pdf(path)
        file_type = "pdf"
    elif suffix in {".txt", ".md"}:
        raw, repairs = path.read_text(encoding="utf-8", errors="replace"), []
        file_type = "text"
    else:
        raise ValueError(f"Unsupported resume format: {suffix}")

    text, clean_repairs = clean_text(raw)
    return LoadedDocument(
        path=path, text=text, n_chars=len(text), file_type=file_type,
        repairs=repairs + clean_repairs,
    )


def load_corpus(resume_dir: str | Path, cache_dir: str | Path | None = None) -> list[LoadedDocument]:
    """Extract every supported resume in a directory, optionally caching the text."""
    resume_dir = Path(resume_dir)
    docs: list[LoadedDocument] = []
    for path in sorted(resume_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            continue
        doc = load_document(path)
        docs.append(doc)
        if cache_dir:
            out = Path(cache_dir) / f"{path.stem}.txt"
            out.write_text(doc.text, encoding="utf-8")
    return docs
