"""LLM-based resume parsing: prompt, cache, concurrency, validation loop.

The extraction contract has four rules, all of which exist because of specific
failure modes seen in the sample corpus:

1. **Extract, never infer.** If the resume does not state a fact, the field is null.
   A search platform that invents sector coverage is worse than one with gaps,
   because a recruiter cannot tell the difference.
2. **Normalise onto controlled vocabularies.** Free-text sectors make filters useless.
   "TMT" -> ["Technology", "Media & Telecom"], "Internet & Interactive Entertainment"
   -> ["Technology", "Media & Telecom"].
3. **Never do arithmetic.** The model must not compute years of experience. It reports
   dates and any self-reported claim; Python computes the truth and flags disagreement.
4. **Report doubt.** Contradictions go into `extraction_notes` and lower
   `extraction_confidence` rather than being silently resolved.

Caching is keyed on a hash of (prompt version + document text), so re-running is free
and deterministic, while any change to the prompt or the source document
automatically invalidates the entry.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import config
import llm_client
from document_loader import LoadedDocument
from pydantic import ValidationError
from schema import CandidateProfile, llm_json_schema

PROMPT_VERSION = "v4"

SYSTEM_PROMPT = """You are a precise resume-parsing engine for the Business Development \
team of a global multi-strategy hedge fund. The team sources junior-to-mid investment \
analyst talent across the US, Europe and Asia-Pacific, across fundamental and \
systematic strategies, and across sectors.

Your job is to convert one resume into a single structured JSON object that conforms \
exactly to the provided schema.

RULES

1. EXTRACT, DO NOT INFER. Only record what the document states or directly implies. \
Use null or an empty list when information is absent. Never invent employers, dates, \
degrees, skills or sectors.

2. NORMALISE onto the schema's controlled vocabularies. Map free text to the closest \
allowed value. Guidance:
   - "TMT", "Tech/Media/Telecom" -> ["Technology", "Media & Telecom"]
   - "Internet", "Software", "Cloud", "Digital Advertising", "Interactive \
Entertainment" -> "Technology" (add "Media & Telecom" when media/streaming/ads are \
explicit)
   - "Pharma", "Biotech", "MedTech", "Diagnostics", "Life Sciences", "Hospitals" -> "Healthcare"
   - "Banks", "Insurance", "Brokers", "Specialty Finance" -> "Financial Services"
   - "Renewables", "Oil & Gas", "Utilities-scale power" -> "Energy"
   - "Infrastructure", "Aerospace", "Logistics", "Manufacturing" -> "Industrials"
   - "Retail", "E-commerce", "Consumer Discretionary", "Alcobev", "Grocery" -> "Consumer"
   - "Structured credit", "High yield", "Investment grade", "Bonds", "Securitisation", \
"Royalty financing" -> "Credit"
   - "Rates", "FX", "Yield curve", "Fixed income derivatives", "Macro" -> "Macro / Rates & FX"
   - Explicitly sector-agnostic or generalist mandates -> "Multi-Sector / Generalist"
   Keep the original wording in `sector_specialisation_detail` so nothing is lost.

3. STRATEGY CLASSIFICATION.
   - "Fundamental": company research, financial modelling, DCF, management meetings, \
long/short stock picking, sell-side equity research.
   - "Systematic / Quantitative": factor models, backtesting, signal research, \
statistical arbitrage, derivatives pricing libraries, algorithmic trading.
   - "Hybrid": material evidence of both.
   - "Unclear": insufficient evidence.

4. DATES. Format every date as YYYY-MM (or YYYY when only a year is given). Use \
"present" for current roles and set is_current true. If tenure is expressed as a \
duration such as "8 years 10 months" with no dates, leave start_date and end_date \
null and put the verbatim string in `duration_stated`.

5. DO NO ARITHMETIC. Never calculate total years of experience. Populate \
`self_reported_years_experience` only when the resume explicitly states a number of \
years of experience. Downstream code computes totals from the dates you extract.

6. FLAG PROBLEMS. `extraction_notes` is a required part of the job, not an optional \
extra. Before returning, re-read the document and record every observation a reviewer \
would want flagged, as a short factual sentence each. Look specifically for:
   - contradictions, and employer names that disagree within one entry
   - impossible dates, and roles or study periods that run concurrently
   - date ranges given as years only, so tenure is approximate
   - missing sections: no email, no phone, no education, no location for the candidate \
(a location stated for an employer is not the candidate's location)
   - malformed contact details
   - self-reported claims that the dates do not support
   - unverifiable performance claims (returns, Sharpe ratios, P&L, rankings, test scores)
   - any classification you had to make on thin evidence, naming the evidence you used
   Do not resolve these yourself and do not omit one because it seems minor. An empty \
`extraction_notes` list asserts that the document is completely unambiguous, which is \
rare; only return an empty list if you genuinely found nothing.

   Calibrate `extraction_confidence` against that list: "low" when the document is \
substantially ambiguous or key sections are missing, "medium" when there are notable \
gaps or several notes, "high" only when the document is clean, complete and produced \
no more than one minor note.

7. PROVENANCE. If a recruiting agency name appears in a header, footer or watermark, \
record it in `source_agency`. It is metadata, not an employer.

Return only the JSON object."""

USER_TEMPLATE = """Parse the resume below into the structured schema.

Source file: {filename}
File type: {file_type}

<resume>
{text}
</resume>"""


@dataclass
class ParseResult:
    filename: str
    profile: CandidateProfile | None
    from_cache: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 0
    latency_s: float = 0.0
    error: str | None = None
    validation_retries: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------- cache
def content_key(text: str) -> str:
    """Hash of prompt version + document text. Any change invalidates the cache."""
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode())
    h.update(SYSTEM_PROMPT.encode())
    h.update(text.encode())
    return h.hexdigest()


def cache_path(stem: str, key: str) -> Path:
    return Path(config.LLM_CACHE_DIR) / f"{stem}.{key[:12]}.json"


def read_cache(stem: str, key: str) -> dict | None:
    p = cache_path(stem, key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def write_cache(stem: str, key: str, payload: dict) -> None:
    cache_path(stem, key).write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------- parsing
def parse_document(doc: LoadedDocument, use_cache: bool = True,
                   force_refresh: bool = False) -> ParseResult:
    """Parse one resume into a validated CandidateProfile."""
    stem = doc.path.stem
    key = content_key(doc.text)

    if use_cache and not force_refresh:
        cached = read_cache(stem, key)
        if cached:
            try:
                profile = CandidateProfile.model_validate(cached["profile"])
                return ParseResult(
                    filename=doc.filename, profile=profile, from_cache=True,
                    prompt_tokens=cached.get("prompt_tokens", 0),
                    completion_tokens=cached.get("completion_tokens", 0),
                    notes=["served from cache"],
                )
            except ValidationError as exc:
                # A stale cache entry must never break the run.
                return _live_parse(doc, key, note=f"cache invalid ({exc.error_count()} errors), re-parsed")

    return _live_parse(doc, key)


def _live_parse(doc: LoadedDocument, key: str, note: str | None = None) -> ParseResult:
    system = SYSTEM_PROMPT
    user = USER_TEMPLATE.format(filename=doc.filename, file_type=doc.file_type, text=doc.text)
    schema = llm_json_schema()

    prompt_tokens = completion_tokens = attempts = 0
    latency = 0.0
    validation_retries = 0
    notes = [note] if note else []
    last_error: str | None = None

    # Two-stage loop: schema-enforced generation, then Pydantic validation. If
    # validation still fails we send the errors back to the model to repair - a
    # cheaper and more reliable fix than discarding the record.
    for validation_attempt in range(2):
        try:
            resp = llm_client.call_llm(system, user, json_schema=schema)
        except llm_client.LLMError as exc:
            return ParseResult(filename=doc.filename, profile=None, error=str(exc))

        prompt_tokens += resp.prompt_tokens
        completion_tokens += resp.completion_tokens
        attempts += resp.attempts
        latency += resp.latency_s

        try:
            data = llm_client.extract_json(resp.text)
            data.setdefault("source_file", doc.filename)
            profile = CandidateProfile.model_validate(data)
        except (llm_client.LLMError, ValidationError) as exc:
            last_error = str(exc)
            validation_retries += 1
            user = (
                USER_TEMPLATE.format(filename=doc.filename, file_type=doc.file_type, text=doc.text)
                + "\n\nYour previous response failed schema validation with these errors. "
                  "Return corrected JSON that satisfies the schema exactly:\n"
                + str(exc)[:1500]
            )
            continue

        write_cache(doc.path.stem, key, {
            "prompt_version": PROMPT_VERSION,
            "content_hash": key,
            "source_file": doc.filename,
            "provider": resp.provider,
            "model": resp.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "profile": profile.model_dump(),
        })
        return ParseResult(
            filename=doc.filename, profile=profile, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, attempts=attempts, latency_s=latency,
            validation_retries=validation_retries, notes=notes,
        )

    return ParseResult(filename=doc.filename, profile=None, error=f"schema validation failed: {last_error}",
                       validation_retries=validation_retries, notes=notes)


def parse_corpus(docs: list[LoadedDocument], use_cache: bool = True, force_refresh: bool = False,
                 max_workers: int | None = None, progress: bool = True) -> list[ParseResult]:
    """Parse many resumes concurrently.

    Concurrency is bounded because provider rate limits, not CPU, are the constraint.
    Cached documents short-circuit without touching the network, so a re-run of an
    unchanged corpus costs nothing.
    """
    workers = max_workers or config.LLM_CONCURRENCY
    results: list[ParseResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(parse_document, d, use_cache, force_refresh): d for d in docs}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if progress:
                status = "cache" if res.from_cache else ("ok" if res.profile else "FAIL")
                print(f"  [{status:5s}] {res.filename}" + (f" - {res.error}" if res.error else ""))
    # Stable output order regardless of completion order.
    order = {d.filename: i for i, d in enumerate(docs)}
    results.sort(key=lambda r: order.get(r.filename, 999))
    return results
