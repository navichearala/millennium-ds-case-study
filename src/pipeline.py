"""End-to-end pipeline: documents -> LLM extraction -> validation -> exports.

Run as a script:
    python src/pipeline.py                # uses cached extractions where available
    python src/pipeline.py --refresh      # force live LLM calls for every resume
    python src/pipeline.py --seed-cache   # rebuild the offline cache from reference extractions

Outputs land in outputs/:
    candidates.json           full nested records (the app's source of truth)
    candidates.csv            flat one-row-per-candidate table for Excel / BI
    candidates_roles.csv      one row per role, for tenure and firm-level analysis
    data_quality_report.csv   one row per flag, for review triage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
import pandas as pd
import parser as resume_parser
import validation
from document_loader import load_corpus
from schema import CandidateProfile, EnrichedCandidate

REFERENCE_FILE = Path(config.DATA_DIR) / "reference_extractions.json"


# ------------------------------------------------------------------ cache seed
def seed_cache_from_reference(verbose: bool = True) -> int:
    """Populate data/llm_cache/ from the committed reference extractions.

    The reference file holds the LLM output for the 10 sample resumes. Seeding lets
    anyone reproduce the notebook and run the app without API credentials, while
    `--refresh` still exercises the live extraction path end to end. Cache keys are
    hashes of prompt version + document text, so if either changes the seeded entries
    are ignored and a live parse is required.
    """
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Reference extractions not found at {REFERENCE_FILE}")

    reference = json.loads(REFERENCE_FILE.read_text())
    docs = load_corpus(config.RESUME_DIR, config.RAW_TEXT_DIR)
    seeded = 0
    for doc in docs:
        payload = reference.get(doc.path.stem)
        if payload is None:
            if verbose:
                print(f"  [skip ] no reference extraction for {doc.filename}")
            continue
        profile = CandidateProfile.model_validate(payload)   # fail fast on schema drift
        key = resume_parser.content_key(doc.text)
        resume_parser.write_cache(doc.path.stem, key, {
            "prompt_version": resume_parser.PROMPT_VERSION,
            "content_hash": key,
            "source_file": doc.filename,
            "provider": "reference",
            "model": "reference-extraction",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "profile": profile.model_dump(),
        })
        seeded += 1
        if verbose:
            print(f"  [seed ] {doc.filename}")
    return seeded


# -------------------------------------------------------------------- exports
def to_flat_row(c: EnrichedCandidate) -> dict:
    """One flat row per candidate for CSV / BI consumption."""
    return {
        "candidate_id": c.candidate_id,
        "full_name": (f"{c.honorific} " if c.honorific else "") + c.full_name,
        "email": c.email,
        "phone": c.phone,
        "location_city": c.location_city,
        "location_country": c.location_country,
        "region": c.region,
        "current_employer": c.current_employer,
        "current_title": c.current_title,
        "is_currently_employed": c.is_currently_employed,
        "strategy_type": c.primary_strategy_type,
        "market_side": c.primary_market_side,
        "firm_type": c.primary_firm_type,
        "seniority_level": c.seniority_level,
        "sectors_covered": "; ".join(c.sectors_covered),
        "n_sectors": len(c.sectors_covered),
        "geographic_markets": "; ".join(c.geographic_markets_covered),
        "years_experience": c.computed_years_experience,
        "years_experience_ex_internships": c.computed_years_excluding_internships,
        "self_reported_years": c.self_reported_years_experience,
        "career_start_year": c.career_start_year,
        "n_roles": c.n_roles,
        "n_employers": c.n_employers,
        "employers": "; ".join(c.employer_list),
        "max_coverage_universe": c.max_coverage_universe,
        "highest_degree_tier": c.highest_degree_tier,
        "highest_degree": c.highest_degree,
        "top_institution": c.education[0].institution if c.education else None,
        "has_cfa": c.has_cfa,
        "cfa_status": c.cfa_status,
        "has_medical_degree": c.has_medical_degree,
        "certifications": "; ".join(c.certifications),
        "programming_languages": "; ".join(c.programming_languages),
        "tools_and_platforms": "; ".join(c.tools_and_platforms),
        "languages_spoken": "; ".join(c.languages_spoken),
        "source_agency": c.source_agency,
        "extraction_confidence": c.extraction_confidence,
        "data_quality_score": c.data_quality_score,
        "n_data_quality_flags": len(c.data_quality_flags),
        "n_career_gaps": len(c.career_gaps),
        "source_file": c.source_file,
    }


def to_role_rows(c: EnrichedCandidate) -> list[dict]:
    rows = []
    for r in c.roles:
        rows.append({
            "candidate_id": c.candidate_id,
            "full_name": c.full_name,
            "employer": r.employer,
            "title": r.title,
            "location": r.location,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "duration_stated": r.duration_stated,
            "is_current": r.is_current,
            "is_internship": r.is_internship,
            "firm_type": r.firm_type,
            "market_side": r.market_side,
            "seniority": r.seniority,
            "strategy_type": r.strategy_type,
            "sectors": "; ".join(r.sectors),
            "coverage_universe_size": r.coverage_universe_size,
            "aum_or_portfolio_size": r.aum_or_portfolio_size,
        })
    return rows


def export(candidates: list[EnrichedCandidate]) -> dict[str, Path]:
    out = Path(config.OUTPUT_DIR)
    paths: dict[str, Path] = {}

    (out / "candidates.json").write_text(
        json.dumps([c.model_dump() for c in candidates], indent=2, default=str)
    )
    paths["json"] = out / "candidates.json"

    pd.DataFrame([to_flat_row(c) for c in candidates]).to_csv(out / "candidates.csv", index=False)
    paths["csv"] = out / "candidates.csv"

    role_rows = [row for c in candidates for row in to_role_rows(c)]
    pd.DataFrame(role_rows).to_csv(out / "candidates_roles.csv", index=False)
    paths["roles_csv"] = out / "candidates_roles.csv"

    flag_rows = [
        {
            "candidate_id": c.candidate_id, "full_name": c.full_name, "source_file": c.source_file,
            "flag_origin": "model" if f.startswith("[model]") else "rule",
            "flag": f.replace("[model] ", ""),
        }
        for c in candidates for f in c.data_quality_flags
    ] + [
        {"candidate_id": c.candidate_id, "full_name": c.full_name, "source_file": c.source_file,
         "flag_origin": "rule", "flag": f"Career gap: {g}"}
        for c in candidates for g in c.career_gaps
    ]
    pd.DataFrame(flag_rows).to_csv(out / "data_quality_report.csv", index=False)
    paths["quality_csv"] = out / "data_quality_report.csv"
    return paths


# --------------------------------------------------------------------- driver
def run(refresh: bool = False, seed: bool = False, verbose: bool = True) -> list[EnrichedCandidate]:
    if seed:
        if verbose:
            print("Seeding offline cache from reference extractions...")
        seed_cache_from_reference(verbose)

    if verbose:
        print(f"\n1. Extracting text from resumes in {config.RESUME_DIR}")
    docs = load_corpus(config.RESUME_DIR, config.RAW_TEXT_DIR)
    for d in docs:
        if verbose:
            note = f" | repairs: {'; '.join(d.repairs)}" if d.repairs else ""
            print(f"  {d.filename:48s} {d.file_type:5s} {d.n_chars:6,d} chars{note}")

    if verbose:
        print(f"\n2. LLM extraction ({len(docs)} documents, cache {'BYPASSED' if refresh else 'enabled'})")
    results = resume_parser.parse_corpus(docs, use_cache=not refresh, force_refresh=refresh, progress=verbose)

    failed = [r for r in results if r.profile is None]
    if failed and verbose:
        print(f"\n  WARNING: {len(failed)} document(s) failed to parse:")
        for r in failed:
            print(f"    {r.filename}: {r.error}")

    profiles = [r.profile for r in results if r.profile]
    tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    if verbose and tokens:
        print(f"  total tokens used this run: {tokens:,}")

    if verbose:
        print(f"\n3. Validation and enrichment ({len(profiles)} profiles)")
    candidates = validation.enrich_all(profiles)
    summary = validation.corpus_quality_summary(candidates)
    if verbose:
        for k, v in summary.items():
            print(f"  {k}: {v}")

    if verbose:
        print("\n4. Exporting")
    paths = export(candidates)
    if verbose:
        for k, p in paths.items():
            print(f"  {k}: {p}")

    # Per-candidate JSON keeps a reviewable artefact next to the aggregate export.
    for c in candidates:
        (Path(config.PARSED_DIR) / f"{c.candidate_id}.json").write_text(
            json.dumps(c.model_dump(), indent=2, default=str)
        )
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse resumes into a searchable candidate dataset")
    ap.add_argument("--refresh", action="store_true", help="ignore cache and call the LLM for every resume")
    ap.add_argument("--seed-cache", action="store_true", help="rebuild the offline cache from reference extractions")
    args = ap.parse_args()
    run(refresh=args.refresh, seed=args.seed_cache)


if __name__ == "__main__":
    main()
