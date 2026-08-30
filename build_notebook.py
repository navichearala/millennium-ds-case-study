"""Build the submission notebook from the live source files.

Generating the notebook programmatically guarantees the code shown to the reviewer is
byte-identical to the code that produced the outputs. Each module is embedded in a
`%%writefile` cell, so the notebook is self-contained: running it top to bottom
recreates the whole project and reproduces every result.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
NB_PATH = ROOT / "2025_ds_case_study_resume_platform.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def module_cell(rel_path: str) -> dict:
    body = (ROOT / rel_path).read_text(encoding="utf-8")
    return code(f"%%writefile {rel_path}\n{body}")


CELLS: list[dict] = []
A = CELLS.append

# ----------------------------------------------------------------- 0. overview
A(md("""
# Data Science Case Study: Candidate Resume Search Platform

**Submitted by Naveen Chearala** &nbsp;|&nbsp; Millennium Business Development team

---

## What was built

An end-to-end pipeline that converts unstructured PDF and Word resumes into a
validated, searchable candidate dataset, plus a Streamlit application that lets BD
users search, score, compare and analyse candidates against a job requisition.

| Requirement from the brief | Where it is met |
| --- | --- |
| Parse resume data from PDF/Word using LLM models via API | Sections 2-4: `document_loader.py`, `llm_client.py`, `parser.py` |
| Create parsed resume data as JSON, CSV, etc. | Section 6: `outputs/candidates.json`, `candidates.csv`, `candidates_roles.csv` |
| Streamlit web app with multi-criteria search | Section 7: `app.py`, 14 filters plus keyword search |
| Visualise candidate distributions and insights | Section 7: six charts plus requisition coverage-gap analysis |
| Design for scalability | Section 9: architecture, cost model and measured throughput |
| Code for parsing and Streamlit in this notebook | Every module is embedded in a `%%writefile` cell below |
| Link to the Streamlit app | Section 7 |
| Discussion of additional features given more time | Section 10 |

## How to run this notebook

```bash
pip install -r requirements.txt
python src/pipeline.py --seed-cache     # build the dataset
streamlit run app.py                    # launch the app
```

Running this notebook top to bottom recreates every source file and reproduces every
output. To exercise the live LLM path, set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and
run section 5 with `refresh=True`.

## Design thesis

Three decisions shape everything below.

**1. The LLM reads; Python decides.** The model extracts only what a document states.
Every derived number - years of experience, career gaps, seniority alignment, match
score - is computed deterministically in Python. Language models are strong readers and
weak arithmeticians, and a hiring dataset needs numbers that are reproducible and
explainable to a hiring manager.

**2. Normalise onto controlled vocabularies, or search silently fails.** Sectors,
regions, strategy types, firm types and seniority levels are enums in the schema and
are handed to the model as part of the JSON Schema it must satisfy. Without this,
"TMT", "Tech/Media" and "Internet & Interactive Entertainment" become three unrelated
filter values and a BD user searching "Technology" misses real candidates.

**3. Flag, never silently fix.** This corpus contains an experience claim that
contradicts its own dates, overlapping employment at a single firm, a bachelor's degree
attributed to a graduate-only school, and bullets crediting employers other than the
one in the heading. Every one is surfaced with the original text intact. In a hiring
context the discrepancy *is* the signal, and auto-correcting it destroys the audit
trail.
"""))

# --------------------------------------------------------------- 1. architecture
A(md("""
---
## 1. Architecture

```
resumes/*.pdf|*.docx
    |
    |   document_loader.py    text extraction, unicode + ligature repair,
    |                         table de-duplication, true reading order
    v
    |   parser.py             prompt construction, content-hash cache,
    |   llm_client.py         provider-agnostic API call, schema enforcement,
    |                         retry with backoff, JSON repair
    v
    |   schema.py             Pydantic CandidateProfile - strict validation
    v
    |   validation.py         derived facts, consistency rules, quality score
    v
outputs/  candidates.json  candidates.csv  candidates_roles.csv  data_quality_report.csv
    |
    v
app.py                       Streamlit requisition search, scoring, comparison, analytics
```

Each stage is independently testable and independently replaceable: swapping the LLM
provider touches only `llm_client.py`, and adding a filter dimension touches only the
schema and the app.
"""))

A(code("""
import sys, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import pandas as pd
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 200)

import config
print("resume directory :", config.RESUME_DIR)
print("output directory :", config.OUTPUT_DIR)
print("as-of date for experience maths :", config.AS_OF_DATE)
print("\\nresumes found:")
for p in sorted(Path(config.RESUME_DIR).iterdir()):
    if p.suffix.lower() in {".pdf", ".docx"}:
        print(f"  {p.name:48s} {p.stat().st_size/1024:6.1f} KB")
"""))

# ------------------------------------------------------------ 2. document layer
A(md("""
---
## 2. Document extraction: the unglamorous part that decides accuracy

Before any model sees a resume, the text has to be correct. Reading the 10 sample files
surfaced three concrete failure modes that naive extraction gets wrong, and each one
would degrade extraction quality silently.

**Ligature loss in PDFs.** `Omar-El-Hassan-202405.pdf` was produced by a pipeline that
encodes "ti" as a single glyph. `pypdf` renders it as U+FFFD, so the raw text reads
`Quan?ta?ve Developer` and `Implementa?on`. Feeding that to a model corrupts job titles
and skills. The loader repairs a vocabulary of finance and engineering terms
deterministically rather than blind-replacing every unknown glyph.

**Table-based Word layouts.** `Viktor-Sharat.docx` and `Zara-Al-Rashid.docx` store the
entire experience section in Word tables with horizontally merged cells. `python-docx`
reports the same text once per underlying grid column, so every row appears up to four
times. Left alone this quadruples token cost and biases the model toward whatever is
repeated. The loader de-duplicates consecutive identical cell values.

**Reading order.** Word documents interleave paragraphs and tables. Iterating all
paragraphs and then all tables scrambles chronology, so the loader walks the document
body in true XML order.

Every repair is recorded on the returned object, so the pipeline can report exactly
what it had to fix.
"""))

A(module_cell("src/document_loader.py"))

A(code("""
from document_loader import load_corpus

docs = load_corpus(config.RESUME_DIR, config.RAW_TEXT_DIR)

pd.DataFrame([{
    "file": d.filename,
    "type": d.file_type,
    "chars": d.n_chars,
    "repairs applied": "; ".join(d.repairs) or "none needed",
} for d in docs])
"""))

A(code("""
# Proof the ligature repair works: the same span before and after cleaning.
import pypdf
from document_loader import clean_text

raw = "\\n".join((p.extract_text() or "") for p in pypdf.PdfReader(
    str(Path(config.RESUME_DIR) / "Omar-El-Hassan-202405.pdf")).pages)
cleaned, _ = clean_text(raw)

print("BEFORE:", raw[:110].replace("\\n", " "))
print("AFTER :", cleaned[:110].replace("\\n", " "))
"""))

# ---------------------------------------------------------------- 3. the schema
A(md("""
---
## 3. The schema is the product

Free-form JSON from a language model is not a dataset. The schema below is handed to
the model as a JSON Schema so it is constrained at generation time, and it is enforced
again with Pydantic afterwards.

Two aspects are worth defending in review.

**Controlled vocabularies.** `Sector`, `Region`, `StrategyType`, `FirmType`,
`MarketSide` and `Seniority` are `Literal` types. The model is instructed to map free
text onto them, with explicit mapping guidance in the prompt. This is what makes a
filter meaningful.

**Claims are separated from facts.** `self_reported_years_experience` captures what the
candidate asserts. `computed_years_experience` is calculated in Python from the
extracted dates. Keeping both lets the platform detect and display the disagreement
instead of picking a winner - and in this corpus one candidate's assertion is off by
almost four years.

`EnrichedCandidate` extends the extracted profile with everything Python computes.
That separation makes it obvious at a glance which fields came from a model and which
came from a rule.
"""))

A(module_cell("src/schema.py"))

A(code("""
from schema import CandidateProfile, EnrichedCandidate

extracted = set(CandidateProfile.model_fields)
enriched = [f for f in EnrichedCandidate.model_fields if f not in extracted]

print(f"Fields extracted by the LLM ({len(extracted)}):")
print("  " + ", ".join(sorted(extracted)))
print(f"\\nFields computed in Python ({len(enriched)}):")
print("  " + ", ".join(enriched))
"""))

# ------------------------------------------------------------- 4. the llm layer
A(md("""
---
## 4. LLM extraction

### Provider portability

The brief allows OpenAI, Anthropic or alternatives. `llm_client.py` resolves the
provider from whichever API key is present, so this notebook runs in any reviewer's
environment without code edits. Both paths enforce the schema at generation time
rather than cleaning up afterwards:

* **OpenAI** - `response_format={"type": "json_schema", ...}`, which makes the API
  reject generations that violate the schema.
* **Anthropic** - a forced tool call whose `input_schema` is the same JSON Schema.

### Resilience and cost control

* Exponential backoff with jitter, so a batch of concurrent parses does not
  synchronise its retries against a rate limit.
* A brace-matching JSON recovery pass for responses wrapped in prose or code fences.
* A second-chance validation loop: if Pydantic still rejects the object, the errors are
  sent back to the model to repair. Cheaper and more reliable than discarding a record.
* Token usage is recorded per call, which is what makes the cost model in section 9 an
  estimate grounded in measurement rather than a guess.

### The extraction contract

The prompt enforces four rules, each written in response to something in this corpus:
extract but never infer; normalise onto the controlled vocabularies; never do
arithmetic; report doubt in `extraction_notes` and lower `extraction_confidence`
rather than resolving a contradiction silently.
"""))

A(module_cell("src/llm_client.py"))
A(module_cell("src/parser.py"))

A(code("""
import parser as resume_parser

print(f"prompt version: {resume_parser.PROMPT_VERSION}")
print("=" * 100)
print(resume_parser.SYSTEM_PROMPT)
"""))

# ------------------------------------------------------------- 5. run the parse
A(md("""
---
## 5. Validation and enrichment rules

`validation.py` is deterministic Python, so every number in the app traces back to a
rule rather than to a model's opinion. It does three things.

**Derived facts.** Years of experience come from the *union* of role date intervals, so
concurrent roles are never double-counted - which matters here, because one candidate
holds an internship and a research assistantship simultaneously. Year-only dates
resolve to January for a start and December for an end, because reading "2016 - 2019"
as ending in January 2019 both understates tenure and manufactures a phantom gap.
Resumes that state tenure as "8 years 10 months" instead of dates are handled
separately.

**Consistency checks.** Self-reported versus computed experience, impossible and
overlapping dates, employment gaps, malformed contact details, duplicate education
entries, degrees attributed to graduate-only schools, and bullets that name a different
employer from their own role heading.

**Judgement about what counts as a red flag.** Gaps are computed on non-internship
roles only, so a student summer is never reported as unemployment. Any gap
substantially covered by a stated study period is annotated as such rather than
presented as unexplained - an MBA is an explanation, not a concern, and a recruiter
should see that distinction without opening the file.
"""))

A(module_cell("src/validation.py"))
A(module_cell("src/pipeline.py"))

A(md("""
### Tests

The LLM step is probabilistic and is covered by schema enforcement plus the evaluation
harness described in section 10. Everything downstream of it is ordinary Python and is
pinned by unit tests, because these rules produce every number a recruiter sees.

Each test corresponds to a real behaviour in this corpus - overlapping roles counted
once, stated durations parsed, ligature damage repaired, an experience claim that
contradicts its own dates, a bachelor's degree at a graduate-only school, a study period
that explains a gap.

Writing these paid for itself immediately: the degree-tier test failed on first run and
exposed a genuine bug. `M.B.B.S` was being classified as a master's degree because the
keyword matcher did not handle internal punctuation, which meant Dr. Zara Al-Rashid's
medical qualification was being under-ranked in search. Both the matcher and the
accent handling were fixed as a result.
"""))

A(module_cell("tests/test_pipeline.py"))

A(code("""
!python -m pytest tests/ -q
"""))

A(md("""
### Running the pipeline

`--seed-cache` writes the committed extraction output (`data/reference_extractions.json`)
into the cache so this notebook is reproducible without credentials. Set
`refresh=True` with an API key exported to bypass the cache entirely and call the live
API for all 10 resumes.
"""))

A(code("""
import pipeline

candidates = pipeline.run(refresh=False, seed=True)
print(f"\\n{len(candidates)} candidates parsed and validated.")
"""))

# ------------------------------------------------------------------ 6. outputs
A(md("""
---
## 6. Parsed output

Four exports, each serving a different consumer: nested JSON for the application, a
flat candidate table for Excel and BI, a role-level table for tenure and firm analysis,
and a flag-level table for review triage.
"""))

A(code("""
flat = pd.read_csv(Path(config.OUTPUT_DIR) / "candidates.csv")
flat[["full_name", "region", "current_employer", "strategy_type", "market_side",
      "seniority_level", "years_experience", "sectors_covered", "highest_degree_tier",
      "has_cfa", "data_quality_score"]]
"""))

A(code("""
# One record in full, to show extraction depth and the provenance fields.
print(json.dumps(candidates[6].model_dump(), indent=2, default=str)[:4000])
"""))

A(code("""
roles = pd.read_csv(Path(config.OUTPUT_DIR) / "candidates_roles.csv")
print(f"{len(roles)} roles extracted across {roles['candidate_id'].nunique()} candidates")
roles[["full_name", "employer", "title", "start_date", "end_date", "duration_stated",
       "firm_type", "market_side", "sectors", "coverage_universe_size"]].head(20)
"""))

A(md("""
### What validation caught

This is the part of the exercise I would most want to discuss. The sample resumes
contain deliberate inconsistencies, and a platform that ingests them without comment
would present a false picture of the talent pool.
"""))

A(code("""
quality = pd.read_csv(Path(config.OUTPUT_DIR) / "data_quality_report.csv")
print(f"{len(quality)} findings across {quality['candidate_id'].nunique()} candidates")
print(f"  raised by validation rules : {(quality['flag_origin'] == 'rule').sum()}")
print(f"  reported by the model      : {(quality['flag_origin'] == 'model').sum()}")

rules = quality[quality["flag_origin"] == "rule"]
for name, group in rules.groupby("full_name"):
    print(f"\\n{name}")
    for f in group["flag"]:
        print(f"  - {f}")
"""))

A(md("""
The findings that matter most for a hiring decision:

| Candidate | Finding | Why it matters |
| --- | --- | --- |
| Priya Nakamura | Profile claims 9 years of healthcare coverage; the dated roles imply 12.7 | The platform reports the computed figure and flags the disagreement rather than choosing one |
| Priya Nakamura | A Jardine Lloyd Thompson role whose bullet says "started my journey as a Lead Analyst at Anand Rathi" | Employer history cannot be taken at face value |
| Omar El-Hassan | A full-time role from May 2022 overlapping an internship at the same firm from April to August 2022 | Chronologically impossible as written |
| Ryan Patel | A Millennium role titled "North53 Capital" whose bullet says the book was "for Vertex Capital" | Two different entities inside one entry |
| Ryan Patel | A Bachelor of Science attributed to Columbia Business School | CBS is graduate-only, so the degree attribution is wrong |
| Vikram Shah | Kellogg listed twice, and a bachelor's degree attributed to Kellogg | Kellogg is graduate-only; also a formatting duplication |
| Marina Silva Costa | A Bain & Company role crediting the launch of "McKinsey's first case competition" | Internal contradiction in the same entry |
| Marcus Chen-Rodriguez | Email `rchen@hotmail` with no top-level domain | Unusable contact detail, and the local part does not match the stated first name |
| Chen Li (Alex) | No role after September 2023 | Currently unplaced, which changes how BD would approach them |
| Viktor Sharat | Tenure given only as durations, with no dates anywhere | Chronology, current status and gaps are genuinely unknowable and are reported as such |
| Priya Nakamura | A "Red Lane Talent Management" watermark | Agency-sourced document, captured as provenance metadata rather than discarded as noise |

Three candidates have no contact details at all, which is a practical sourcing
obstacle worth surfacing before someone tries to reach out.
"""))

A(code("""
# Corpus-level view: the shape of the talent pool the BD team actually has.
import validation
print(json.dumps(validation.corpus_quality_summary(candidates), indent=2))

print("\\nRegion x strategy:")
print(pd.crosstab(flat["region"], flat["strategy_type"]))
print("\\nSector coverage (candidates per sector):")
print(flat["sectors_covered"].str.split("; ").explode().value_counts())
print("\\nExperience:")
print(flat["years_experience"].describe().round(1))
"""))

# ---------------------------------------------------------------- 7. the app
A(md("""
---
## 7. The Streamlit application

**Link:** the app runs locally at **http://localhost:8501** after
`streamlit run app.py`. It is published locally as the brief specifies; no candidate
data leaves the machine, which is the right default for resume data even when it is
synthetic. Deploying it to an internal host is a configuration change, not a code
change - see section 9.

### Interface design

BD does not browse candidates, it fills mandates. So the app is organised around a
**requisition** - region, investment approach, sector coverage and an experience band -
with five preset mandates modelled on the search dimensions named in the brief.
Requisitions are deep-linkable (`?req=2`) so a recruiter can send a colleague the exact
view rather than a description of which filters to set.

**Search and filtering.** Fourteen filters: region, investment approach, sector
coverage with any/all logic, experience band, seniority, firm type, market side,
highest degree, language, CFA, medical degree, currently-employed, minimum data-quality
score, and comma-separated keyword search across the full resume text.

**Transparent scoring.** The match score is a visible weighted sum over five
components - sector fit, region, strategy, experience and credentials - each a
documented ratio, with the weights exposed as sliders and a per-candidate contribution
chart. A hiring manager can always be told exactly why one candidate ranks above
another. This matters beyond aesthetics: an opaque ranking in a hiring context is a
liability, and a score nobody can explain will not be trusted or used.

Two deliberate softening choices. Experience bands drive the score with linear decay
outside the band rather than acting as a hard cut, because excluding someone for being
six months outside a range loses good people; a checkbox makes it a hard filter when a
recruiter really wants one. And when no requisition criteria are set, the app ranks by
experience and shows data quality instead of a score, because with every dimension set
to "Any" every candidate scores near-perfectly and the ranking would be meaningless.

**Four views.** Search results (cards or table, with CSV shortlist export), side-by-side
comparison with a normalised radar chart, talent-pool analytics, and a data-quality
review tab.

**Performance.** Data loads once through `st.cache_data`, because Streamlit re-runs the
whole script on every widget change. Filtering is a vectorised boolean mask over a
pandas frame rather than per-row Python, so the interaction model holds as the corpus
grows.
"""))

A(md("""
### Search results and match scoring

![Search results](outputs/shot_01_results.png)

Scoring against the "US Healthcare Fundamental Analyst" mandate. Ryan Patel scores 92
and Marcus Chen-Rodriguez 91; both are visibly tagged with their data-quality flag
counts, so a recruiter knows to check before acting.

### Side-by-side comparison

![Comparison](outputs/shot_02_compare.png)

### Talent pool analytics

![Insights](outputs/shot_03_insights.png)

Sector coverage by region, fundamental versus systematic mix, experience distribution,
current firm type, coverage universe against experience, and tooling frequency - plus a
coverage-gap table showing where the pipeline is thin for the sectors the current
requisition needs.

### Data quality review

![Data quality](outputs/shot_04_quality.png)

Every finding, labelled by whether a validation rule or the extraction model raised it,
filterable and exportable.
"""))

A(module_cell("app.py"))

A(md("""
Launch it with:

```bash
streamlit run app.py
```
"""))

# ------------------------------------------------------- 8. insights for BD
A(md("""
---
## 8. What the data says about this pool

Ten candidates is too small for statistical claims, but the composition is already
actionable, and these are the observations I would bring to a BD stakeholder.

**The pool is heavily fundamental and heavily healthcare.** Eight of ten are
fundamental investors; only Chen Li and Omar El-Hassan are systematic or quantitative.
Healthcare is the deepest sector with seven candidates, and three of the four
Asia-Pacific candidates are healthcare sell-side research analysts. If the open requisitions skew systematic or skew technology outside
the US, this pipeline does not cover them.

**Regional depth is uneven by sector.** US candidates cluster in TMT, consumer and
generalist mandates. APAC candidates cluster almost entirely in healthcare sell-side
research. Europe has two candidates covering entirely different things - one
fundamental equity generalist and one rates-and-credit quant developer. The
coverage-gap table in the app makes this concrete per requisition.

**Two profiles are genuinely differentiated.** Dr. Zara Al-Rashid pairs an MBBS with a
finance PGDM and ten years of pharma research - a clinical-depth profile that is hard
to source. Viktor Sharat pairs a biotechnology and bioinformatics M.Tech and a machine
learning publication with a decade of sell-side coverage. Both are sourcing assets a
keyword search on job titles would never surface, which is an argument for extracting
education and publications rather than just employment.

**One candidate is an obvious priority call.** Ryan Patel previously worked at
Millennium, is currently running a fundamental long/short book, and covers consumer,
TMT and healthcare. Prior familiarity with the platform is directly relevant context,
and the platform surfaces it because employers are extracted as structured fields.

**Two candidates are not currently placed.** Chen Li has no role recorded after
September 2023, and Viktor Sharat's document gives no dates at all. Both change how BD
would prioritise outreach, and neither is visible from a job title.
"""))

# ------------------------------------------------------------ 9. scalability
A(md("""
---
## 9. Designing for scale

The brief asks for a design that handles large volumes. The current implementation
already contains the parts that matter at scale; the rest is a described migration path
rather than speculation.

### Already built for it

| Concern | Implementation |
| --- | --- |
| Redundant LLM spend | Content-hash cache keyed on prompt version + document text; a re-run of an unchanged corpus makes zero API calls |
| Throughput | `ThreadPoolExecutor` with bounded concurrency, since provider rate limits rather than CPU are the constraint |
| Transient failure | Exponential backoff with jitter, plus a validation-repair round trip |
| Partial failure | One bad document cannot fail the batch; failures are collected and reported |
| Schema drift | Pydantic validation on every record, including cache reads, so a stale cache entry re-parses instead of breaking |
| UI responsiveness | `st.cache_data` on load, vectorised mask filtering, no per-row Python in the hot path |
| Reproducibility | Fixed `AS_OF_DATE`, temperature 0, versioned prompt |
| Regression safety | 25 unit tests over the validation and cleaning rules |

### Measured cost model

At roughly 1,100 input and 900 output tokens per resume (measured on this corpus), a
small frontier model at approximately $0.15 per million input and $0.60 per million
output tokens costs about **$0.0007 per resume**, or roughly **$700 per million
resumes** - and near zero for re-processing, because of the cache. Extraction cost is
not the binding constraint; document acquisition and review capacity are.

### Migration path from 10 to 1,000,000

**Ingestion.** Replace directory scanning with object storage plus an event queue (S3
plus SQS, or equivalent). Each message carries one document; workers scale
horizontally. Deduplicate on content hash before spending a single token - resume
corpora are full of near-duplicates from multiple agencies submitting the same person.

**Extraction.** Route by document complexity: a cheap model for clean single-column
resumes, a stronger model only for documents where confidence is low or validation
fails. Use provider batch APIs for the backlog, since a bulk backfill is not latency
sensitive and batch pricing is materially cheaper. Add OCR for scanned PDFs, which this
corpus does not contain but a real pipeline will hit within its first thousand files.

**Storage.** Move from JSON files to Postgres: a `candidates` table plus `roles`,
`education` and `flags` tables, with GIN indexes on the array columns for sector and
skill filtering, and a `tsvector` column for keyword search. Retain the raw text and
the model response for every record, because auditability is a hard requirement when
the output influences hiring.

**Search.** At a few hundred thousand records, structured filtering belongs in the
database rather than in pandas, and the app becomes a thin query layer over indexed
columns. Add embeddings over the highlight text for genuine semantic search - "someone
who has run a factor model on Asian equities" is a query no keyword index answers well,
and it is exactly how a BD user thinks. Store vectors in pgvector alongside the
structured record so a single query can combine both.

**Incremental processing.** Resumes are re-submitted with updates. Version records by
content hash, keep history, and diff versions so a recruiter can see what changed since
last contact.

**Operations.** Track extraction confidence, validation flag rates and per-field null
rates as monitored metrics. A jump in the null rate for `start_date` is how you find out
a new agency template broke the parser, before a recruiter finds out by getting bad
search results.
"""))

# -------------------------------------------------------- 10. more time / risks
A(md("""
---
## 10. What I would build with more time

Ordered by value per hour of effort.

**1. An evaluation harness (highest priority).** The gap between this and a production
system is measurement. I would hand-label a gold-standard set of 30-50 resumes, then
report per-field precision and recall for every extracted field on every prompt or model
change. Without that, "the extraction is accurate" is an assertion. This is the first
thing I would build with another day, because it turns every subsequent change from a
guess into a measurement.

**2. Semantic search over experience.** Embed role highlights and support natural
language queries - "covered digital health from the buy side in Asia" - blended with
structured filters. This is the single largest usability gain for a BD user.

**3. Requisition-to-candidate matching from a job description.** Paste a real
requisition, have the model extract the structured criteria, and score the whole pool
against it automatically. That closes the loop from the actual artefact BD works from.

**4. Field-level confidence and targeted human review.** Confidence per field rather
than per document, with a review queue that surfaces only low-confidence fields. Review
capacity, not extraction, is the real bottleneck at volume.

**5. Entity resolution.** "J.P. Morgan", "J.P.Mogan", "JPMorgan Chase" and "J.P. Morgan
Asset Management" appear across this corpus and are partly the same institution, partly
not. A firm master table with aliases, plus firm tier and strategy metadata, would let
BD filter by pedigree rather than by string. Candidate-level deduplication belongs here
too, since the same person arrives from multiple agencies.

**6. Bias and fairness controls.** Any tool that ranks people needs this before it is
used in anger: excluding name, nationality, age proxies and gender-correlated signals
from scoring; auditing score distributions across demographic proxies; logging every
shortlist for review. The current scoring uses only sector, region, strategy, experience
and credentials, which is deliberate, but it has not been audited and I would not claim
otherwise.

**7. Access control, PII handling and audit logging.** Real resumes are personal data.
Encryption at rest, role-based access, retention policies aligned to GDPR and
equivalents, redaction of contact details until a recruiter has a legitimate reason to
see them, and an immutable log of who viewed and exported what.

**8. Pipeline hardening.** OCR for scanned documents, language detection and non-English
extraction, fixture resumes for each failure mode on top of the existing unit tests, and
CI that fails on a regression in extraction quality.

---

## 11. Honest limitations

Stated explicitly, because a reviewer will find them anyway and I would rather discuss
them directly.

* **No accuracy measurement.** Extraction correctness was verified by reading all 10
  resumes against the output, not by a labelled evaluation with reported metrics. On a
  corpus of ten that is feasible; it does not generalise, and item 1 above is the fix.
* **Ten records is not a talent pool.** The analytics in section 8 are directional
  observations, not statistics. Several charts would look different with a hundred
  records and should not be over-read.
* **Scoring weights are judgement, not evidence.** The default 35/20/20/15/10 split
  encodes my assumption that sector fit matters most for these mandates. It has not
  been validated against hiring outcomes, which is precisely why the weights are
  exposed as sliders rather than hard-coded and hidden.
* **Some classifications are genuinely arguable.** Chen Li does quantitative factor
  work but also fundamental valuation as an intern, and is classified systematic. The
  `sector_specialisation_detail` field preserves the verbatim evidence so a reviewer can
  disagree with the label without losing the underlying information.
* **Seniority titles are not comparable across firms.** "Associate" means different
  things at Goldman Sachs, Bain and Apollo. Seniority therefore informs the score but
  never gates a search by default.
* **The employer-contradiction check uses a curated firm list.** It reliably catches
  what is in this corpus and would need the entity-resolution work in item 5 to
  generalise.

## 12. How AI was used, and how the output was checked

The brief permits AI, so the honest account: an LLM performs the extraction step, which
is the point of the exercise, and AI assistance was used while writing this code.

What matters is the control structure around it. The model is constrained by a JSON
Schema at generation time and validated by Pydantic afterwards. It is forbidden from
doing arithmetic, so every derived number comes from deterministic Python. Its output
is checked against a dozen consistency rules that compare its extraction to itself and
to the source dates. It is required to report ambiguity rather than resolve it. The
extracted text it received is retained verbatim and is one click away in the app. And
every one of the ten source resumes was read end to end by hand and compared against
the extracted record - which is how the planted contradictions in section 6 were found,
and how the validation rules that catch them were designed.
"""))

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
            "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH} - {len(CELLS)} cells "
      f"({sum(1 for c in CELLS if c['cell_type'] == 'code')} code, "
      f"{sum(1 for c in CELLS if c['cell_type'] == 'markdown')} markdown)")
