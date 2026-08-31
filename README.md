# Candidate Resume Search Platform

Data science case study submission for the Millennium Business Development team.

An LLM-powered pipeline that turns unstructured PDF and Word resumes into a validated,
searchable candidate dataset, plus a Streamlit application that lets BD users search,
score, compare and analyse candidates against a job requisition.

> **Note on scope.** The Streamlit app (`app.py`) is the deliverable the case study asked
> for and is published locally as specified. `web-mirror/` is a supplementary static
> browser port of the same interface — identical scoring maths and filter semantics, no
> Python required — built so the platform can be demonstrated without a local
> environment. It is not a replacement for the submission.

---

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. rebuild the dataset from the committed LLM responses (no API key needed)
python src/pipeline.py

# 3. run the app
streamlit run app.py          # http://localhost:8501

# optional: run the test suite
python -m pytest tests/ -q    # 30 tests
```

To re-run the extraction live against the API:

```bash
pip install -r requirements-llm.txt
export OPENAI_API_KEY=sk-...            # or ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_MODEL=gpt-4.1-mini        # optional
python src/pipeline.py --refresh        # bypasses cache, calls the API for all 10 resumes
```

## Extraction provenance

The committed dataset in `outputs/` was produced by **live OpenAI API calls**, not by
rule-based parsing. Every file in `data/llm_cache/` records the provider, model and real
token counts for its document, so any record can be traced back to the call that made it.

| Live run (`python src/pipeline.py --refresh`) | Value |
| --- | --- |
| Documents parsed via API | 10 / 10, zero failures |
| Model | `gpt-4.1-mini` (`gpt-4.1-mini-2025-04-14`) |
| Structured output | `response_format: json_schema`, temperature 0 |
| Prompt version | `v4` (`PROMPT_VERSION` in `src/parser.py`) |
| Input / output tokens | 35,661 / 13,431 (mean 3,566 / 1,343 per resume) |
| Schema-repair retries needed | 0 |
| Cost at $0.40 / $1.60 per 1M tokens | **$0.0358 total, $0.0036 per resume** |
| Findings raised | 68 reported by the model, 25 by validation rules |
| Mean data-quality score | 0.644 |

The provider is selected automatically from whichever key is present. An
OpenAI-compatible endpoint works by also setting `OPENAI_BASE_URL`.

---

## What it does

```
resumes/*.pdf|docx
   |
   |  document_loader.py     text extraction, unicode/ligature repair,
   |                         table de-duplication, true reading order
   v
   |  parser.py + llm_client.py   LLM extraction with a strict JSON Schema,
   |                              retries, JSON repair, content-hash caching
   v
   |  schema.py              Pydantic validation (CandidateProfile)
   v
   |  validation.py          derived facts + consistency rules + quality score
   v
outputs/candidates.json | candidates.csv | candidates_roles.csv | data_quality_report.csv
   |
   v
app.py                     Streamlit requisition search, scoring, comparison, analytics
```

## Deliverables

| File | Contents |
| --- | --- |
| `2025_ds_case_study_resume_platform.ipynb` | The submitted notebook: all parsing and app code, results, analysis, scalability design |
| `app.py` | Streamlit application |
| `src/` | Pipeline modules (`document_loader`, `schema`, `llm_client`, `parser`, `validation`, `pipeline`, `config`) |
| `outputs/candidates.json` | Full nested records, one object per candidate |
| `outputs/candidates.csv` | Flat one-row-per-candidate table |
| `outputs/candidates_roles.csv` | One row per role, for tenure and firm-level analysis |
| `outputs/data_quality_report.csv` | One row per validation finding |
| `data/reference_extractions.json` | The live LLM extraction output for all 10 resumes, in one file; seeds the offline cache |
| `data/raw_text/` | Text as the model received it, for auditability |
| `tests/test_pipeline.py` | 30 unit tests over the validation and document-cleaning rules |
| `build_notebook.py` | Generates the submission notebook from the live source files, so the code shown to a reviewer is byte-identical to the code that produced the outputs |
| `tools/capture_screenshots.py` | Regenerates the app screenshots from a running instance, so documentation cannot drift from the data |
| `tools/build_web_mirror_data.py` | Rebuilds the browser mirror's embedded dataset from the pipeline output |

## Design decisions

**The LLM reads; Python decides.** The model extracts only what the document states.
Every derived number - years of experience, career gaps, seniority alignment, match
scores - is computed deterministically in Python. LLMs are strong readers and weak
arithmeticians, and a hiring dataset needs numbers that are reproducible and
explainable.

**Controlled vocabularies, not free text.** Sectors, regions, strategy types, firm
types and seniority are `Literal` enums in the schema and are handed to the model as
part of the JSON Schema. Without this, "TMT", "Tech/Media" and "Internet" become three
unrelated filter values and search silently fails.

**Flag, never silently fix.** The corpus contains contradictions: an experience claim
that disagrees with its own dates, overlapping employment at one firm, a bachelor's
degree attributed to a graduate-only school, a bullet crediting a different employer.
All are surfaced with the original text intact. In a hiring context the discrepancy is
often the signal, and auto-correcting it destroys the audit trail.

**Decision support, not automated screening.** The match score is a visible weighted
sum with a per-component breakdown and adjustable weights, no candidate is ever hidden
without the filter that removed them being visible, and the source resume text is one
click from every record.

**Caching keyed on content.** Cache keys hash the prompt version plus the document
text, so re-runs are free and deterministic while any change to either automatically
forces a fresh extraction.

## Reproducibility

`data/llm_cache/` holds the actual API responses from the live run, keyed by a hash of
prompt version plus document text, with the provider, model and token counts attached.
`data/reference_extractions.json` is the same 10 profiles in one file, and `--seed-cache`
rebuilds the cache from it. Because the responses are committed, `python src/pipeline.py`
reproduces the exact dataset with no credentials and no API spend, while `--refresh`
ignores the cache and re-runs the live API path. All experience calculations use a fixed
`AS_OF_DATE` (configurable in `src/config.py`) so results do not drift with the calendar.

## Repository layout

```
2025_ds_case_study_resume_platform.ipynb   submission notebook (executed, with outputs)
app.py                                     Streamlit application
src/                                       pipeline modules
tests/                                     30 unit tests over the deterministic rules
resumes/                                   the 10 sample resumes
data/reference_extractions.json            live extraction output (seeds the cache)
data/raw_text/                             text as the model received it, for auditability
outputs/                                   JSON + CSV exports and app screenshots
web-mirror/                                static browser port of the app (no backend)
build_notebook.py                          regenerates the notebook from the source files
tools/capture_screenshots.py               regenerates the app screenshots from a running app
tools/build_web_mirror_data.py             rebuilds the browser mirror's embedded dataset
```

## Running the browser mirror

No build step and no dependencies — it is plain HTML, CSS and ES modules with the parsed
data baked into `web-mirror/js/data.js`:

```bash
cd web-mirror && python -m http.server 8790   # then open http://localhost:8790
```

Regenerate its data bundle after re-running the pipeline:

```bash
python -c "
import json, pathlib
cands = json.loads(pathlib.Path('outputs/candidates.json').read_text())
raw = {c['candidate_id']: pathlib.Path('data/raw_text', pathlib.Path(c['source_file']).stem + '.txt').read_text()
       for c in cands}
pathlib.Path('web-mirror/js/data.js').write_text(
    'window.PLATFORM_DATA = ' + json.dumps({'candidates': cands, 'raw_text': raw}) + ';\n')
"
```

## A note on the sample data

`resumes/` contains the 10 synthetic resumes supplied with the case study. They are
made up, but they are shaped like real candidate records — names, email addresses and
phone numbers. Keep this repository private, and treat the parsed exports in `outputs/`
the same way.

## Requirements

Python 3.10+. See `requirements.txt`. The pipeline reproduces the committed dataset
without API credentials by replaying the cached responses; install `requirements-llm.txt`
and set a key in `.env` (see `.env.example`) to re-run the live extraction path with
`python src/pipeline.py --refresh`.
