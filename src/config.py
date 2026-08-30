"""Central configuration for the candidate resume search platform.

Keeping paths and tunables in one module means the notebook, the CLI pipeline and
the Streamlit app all agree on where data lives.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUME_DIR = PROJECT_ROOT / "resumes"
DATA_DIR = PROJECT_ROOT / "data"
RAW_TEXT_DIR = DATA_DIR / "raw_text"          # extracted plain text, one file per resume
LLM_CACHE_DIR = DATA_DIR / "llm_cache"        # raw LLM responses keyed by content hash
PARSED_DIR = DATA_DIR / "parsed"              # validated per-candidate JSON
OUTPUT_DIR = PROJECT_ROOT / "outputs"         # consolidated JSON / CSV exports

CANDIDATES_JSON = OUTPUT_DIR / "candidates.json"
CANDIDATES_CSV = OUTPUT_DIR / "candidates.csv"
QUALITY_REPORT_CSV = OUTPUT_DIR / "data_quality_report.csv"

for _d in (RAW_TEXT_DIR, LLM_CACHE_DIR, PARSED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- llm settings
# Provider is resolved at runtime from whichever key is present, so the same code
# runs against OpenAI, Anthropic or an OpenAI-compatible endpoint without edits.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")     # auto | openai | anthropic
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

LLM_TEMPERATURE = 0.0        # deterministic extraction: we want reproducibility, not creativity
LLM_MAX_TOKENS = 4096
LLM_MAX_ATTEMPTS = 4         # retries cover rate limits and malformed JSON
LLM_CONCURRENCY = 5          # parallel resume parses; raise for large batches

# Reference date for all "years of experience" maths so results are reproducible
# regardless of when the notebook is re-run.
AS_OF_DATE = os.getenv("AS_OF_DATE", "2026-08-29")

# ---------------------------------------------------------------- taxonomies
# Controlled vocabularies. The LLM is instructed to map free text onto these, which
# is what makes filtering possible: "TMT", "Tech/Media", "Internet" all collapse to
# a single canonical sector value.
REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"]

SECTORS = [
    "Technology", "Media & Telecom", "Healthcare", "Financial Services", "Energy",
    "Industrials", "Consumer", "Real Estate", "Utilities", "Materials",
    "Credit", "Macro / Rates & FX", "Multi-Sector / Generalist",
]

STRATEGY_TYPES = ["Fundamental", "Systematic / Quantitative", "Hybrid", "Unclear"]

FIRM_TYPES = [
    "Hedge Fund", "Asset Manager", "Investment Bank - Sell-Side Research",
    "Investment Bank - Banking / Markets", "Private Equity / Venture Capital",
    "Commercial / Corporate Bank", "Consulting", "Corporate / Industry",
    "Academic / Research", "Other",
]

SENIORITY_LEVELS = ["Intern", "Analyst", "Senior Analyst", "Associate", "Lead Analyst", "Portfolio Manager", "Other"]
