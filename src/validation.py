"""Post-extraction validation and enrichment.

This is the layer that makes LLM output trustworthy enough to search on. Everything
here is deterministic Python, so every number in the app can be traced back to a rule
rather than to a model's opinion.

Three groups of work:

* **Derived facts** - years of experience from merged date intervals (so overlapping
  roles are not double-counted), career start, employer count, highest degree tier.
* **Consistency checks** - self-reported vs computed experience, impossible or
  overlapping dates, employment gaps, malformed contact details, employer names that
  contradict themselves inside one entry, degrees attributed to the wrong school.
* **A quality score** - a single 0-1 number the app can sort and filter on, so a
  recruiter can choose to review only clean records, or deliberately inspect the
  messy ones.

Flags are surfaced, never auto-corrected. Silently "fixing" a resume destroys the
audit trail, and in a hiring context the discrepancy itself is often the signal.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

import config
from schema import CandidateProfile, EnrichedCandidate

# --------------------------------------------------------------------- helpers
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

# Degree keywords are matched against a punctuation-stripped form of the degree string,
# because resumes write the same qualification as "M.B.B.S", "MBBS" and "M.B.B.S.".
# Matching is on word boundaries so a short abbreviation such as "MD" cannot fire inside
# an unrelated token.
DEGREE_TIERS = {
    "Doctorate / Medical": [
        "phd", "doctor of philosophy", "doctorate", "dphil", "mbbs", "md", "mbchb", "dds", "dvm",
    ],
    "Master's / MBA": [
        "mba", "master", "masters", "msc", "ms", "ma", "mtech", "mcom", "meng", "mphil",
        "pgdm", "post graduate diploma", "llm", "ingenieur", "ingénieur", "mfin", "mfe",
    ],
    "Bachelor's": [
        "bachelor", "bachelors", "bsc", "bs", "ba", "bba", "btech", "bms", "bcom",
        "beng", "undergraduate", "ab",
    ],
}
TIER_RANK = {"Doctorate / Medical": 3, "Master's / MBA": 2, "Bachelor's": 1, "Unknown": 0}

# Graduate-only business schools. A bachelor's degree attributed to one of these is a
# resume error worth flagging, and it is exactly the kind of detail a BD reviewer
# would want to verify before a screen.
GRADUATE_ONLY_SCHOOLS = [
    "kellogg", "sloan school", "columbia business school", "wharton mba",
    "booth school", "harvard business school", "stanford graduate school of business",
]

# Institutions and firms named often enough in this corpus to make an internal
# contradiction detectable, e.g. a Bain role whose bullet credits McKinsey.
KNOWN_FIRMS = [
    "mckinsey", "bain", "boston consulting", "goldman sachs", "morgan stanley", "j.p. morgan",
    "jp morgan", "jpmorgan", "credit suisse", "citi", "barclays", "ubs", "deutsche bank",
    "anand rathi", "kotak", "icici", "axis", "leerink", "william blair", "fidelity",
    "vanguard", "coatue", "cinctive", "apollo", "blackstone", "millennium", "bnp paribas",
    "societe generale", "société générale", "magnetar", "pwc", "centrum", "jardine lloyd thompson",
    "transparent value", "bank of china", "meridian", "north53", "vertex capital", "prism",
]


def parse_ym(value: str | None, is_end: bool = False) -> tuple[int, int] | None:
    """Parse 'YYYY-MM', 'YYYY' or 'present' into a (year, month) tuple.

    Year-only dates resolve to January for a start and December for an end. A resume
    that says "2016 - 2019" means roughly four calendar years; treating the end as
    January 2019 would both understate tenure and manufacture a phantom gap.
    """
    if not value:
        return None
    value = str(value).strip().lower()
    if value in {"present", "current", "now"}:
        y, m, _ = config.AS_OF_DATE.split("-")
        return int(y), int(m)
    m = re.match(r"^(\d{4})-(\d{1,2})", value)
    if m:
        month = min(max(int(m.group(2)), 1), 12)
        return int(m.group(1)), month
    m = re.match(r"^(\d{4})$", value)
    if m:
        return int(m.group(1)), (12 if is_end else 1)
    return None


def to_months(ym: tuple[int, int]) -> int:
    return ym[0] * 12 + (ym[1] - 1)


def from_months(total: int) -> str:
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def parse_duration_to_months(text: str | None) -> int | None:
    """Convert '8 years 10 months' / '10 months' / '2 months' into a month count."""
    if not text:
        return None
    t = text.lower()
    years = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", t)
    months = re.search(r"(\d+)\s*months?", t)
    weeks = re.search(r"(\d+)\s*weeks?", t)
    if not (years or months or weeks):
        return None
    total = 0.0
    if years:
        total += float(years.group(1)) * 12
    if months:
        total += float(months.group(1))
    if weeks:
        total += float(weeks.group(1)) / 4.345
    return max(int(round(total)), 1)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of [start, end) month intervals, so overlapping roles count once."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def classify_degree(degree: str | None) -> str:
    """Map a free-text degree string onto a comparable tier.

    Tiers are checked highest-first, so a combined 'MBBS, MBA' resolves to the doctorate
    tier rather than the master's tier.
    """
    if not degree:
        return "Unknown"
    # Strip accents so "Diplome" and "Diplôme" are the same word.
    lowered = "".join(
        ch for ch in unicodedata.normalize("NFKD", degree.lower())
        if not unicodedata.combining(ch)
    )
    # Two views of the same string. Tokens have internal punctuation removed, so the
    # single token "m.b.b.s" becomes "mbbs" and matches by exact equality - which is
    # safer than a substring search, because "md" must not match inside another word.
    tokens = {re.sub(r"[^a-z0-9]", "", t) for t in re.split(r"[\s/,;()\-'\u2019]+", lowered)}
    phrase = re.sub(r"[^a-z0-9]+", " ", lowered).strip()

    for tier, keys in DEGREE_TIERS.items():
        for key in keys:
            if " " in key:
                if key in phrase:
                    return tier
            elif key in tokens:
                return tier
    return "Unknown"


# ------------------------------------------------------------------ enrichment
def enrich(profile: CandidateProfile) -> EnrichedCandidate:
    """Compute derived fields and run every consistency check."""
    flags: list[str] = []
    now_months = to_months(parse_ym("present"))  # type: ignore[arg-type]

    # ---------------- experience from dates
    all_intervals: list[tuple[int, int]] = []
    non_intern_intervals: list[tuple[int, int]] = []
    undated_months = 0
    undated_intern_months = 0
    starts: list[int] = []

    for role in profile.roles:
        start = parse_ym(role.start_date)
        end = parse_ym(role.end_date, is_end=True) or (parse_ym("present") if role.is_current else None)

        if start and end:
            s, e = to_months(start), to_months(end)
            if e < s:
                flags.append(
                    f"Impossible dates at {role.employer}: end ({role.end_date}) precedes start ({role.start_date})"
                )
                s, e = e, s
            if s > now_months:
                flags.append(f"Future start date at {role.employer}: {role.start_date}")
            starts.append(s)
            all_intervals.append((s, e))
            if not role.is_internship:
                non_intern_intervals.append((s, e))
        else:
            # Resume gave tenure instead of dates (common in Indian sell-side formats).
            months = parse_duration_to_months(role.duration_stated)
            if months:
                undated_months += months
                if not role.is_internship:
                    undated_intern_months += months
            elif role.employer:
                flags.append(f"No usable dates for role at {role.employer}")

    merged_all = merge_intervals(all_intervals)
    merged_non_intern = merge_intervals(non_intern_intervals)
    months_all = sum(e - s for s, e in merged_all) + undated_months
    months_non_intern = sum(e - s for s, e in merged_non_intern) + undated_intern_months

    computed_years = round(months_all / 12, 1)
    computed_years_ex_intern = round(months_non_intern / 12, 1)

    # ---------------- overlapping roles
    dated_roles = [
        (r, to_months(parse_ym(r.start_date)), to_months(parse_ym(r.end_date, is_end=True) or parse_ym("present")))  # type: ignore[arg-type]
        for r in profile.roles
        if parse_ym(r.start_date) and (parse_ym(r.end_date, is_end=True) or r.is_current)
    ]
    for i in range(len(dated_roles)):
        for j in range(i + 1, len(dated_roles)):
            (ra, sa, ea), (rb, sb, eb) = dated_roles[i], dated_roles[j]
            overlap = min(ea, eb) - max(sa, sb)
            if overlap > 1:  # tolerate one month of rounding at role boundaries
                same_employer = ra.employer.strip().lower() == rb.employer.strip().lower()
                kind = "same employer" if same_employer else "different employers"
                flags.append(
                    f"Overlapping roles ({kind}, ~{overlap} months): "
                    f"{ra.employer} [{ra.start_date}-{ra.end_date or 'present'}] and "
                    f"{rb.employer} [{rb.start_date}-{rb.end_date or 'present'}]"
                )

    # ---------------- employment gaps and current status
    #
    # Gaps are computed on non-internship roles only, so a student summer between two
    # academic years is never reported as unemployment. Any gap substantially covered by
    # a stated study period is annotated as such rather than presented as unexplained -
    # an MBA is an explanation, not a red flag, and a recruiter should see the difference
    # at a glance.
    education_windows: list[tuple[int, int, str]] = [
        (edu.start_year * 12, (edu.end_year + 1) * 12, f"{edu.degree or 'study'} at {edu.institution}")
        for edu in profile.education
        if edu.start_year and edu.end_year
    ]

    def explained_by_study(gap_start: int, gap_end: int) -> str | None:
        span = max(gap_end - gap_start, 1)
        for w_start, w_end, label in education_windows:
            covered = min(gap_end, w_end) - max(gap_start, w_start)
            if covered / span >= 0.6:
                return label
        return None

    gaps: list[str] = []
    gap_basis = merged_non_intern or merged_all
    for (_s1, e1), (s2, _e2) in zip(gap_basis, gap_basis[1:]):
        gap = s2 - e1
        if gap >= 6:
            reason = explained_by_study(e1, s2)
            suffix = f" - overlaps {reason}" if reason else " - unexplained"
            gaps.append(f"{gap} month gap between {from_months(e1)} and {from_months(s2)}{suffix}")

    latest_end = max((e for _s, e in merged_all), default=None)
    months_since_last = None
    is_employed = any(r.is_current for r in profile.roles)
    if latest_end is not None:
        months_since_last = max(now_months - latest_end, 0)
        if not is_employed and months_since_last >= 6:
            gaps.append(f"Not currently employed - {months_since_last} months since {from_months(latest_end)}")

    # ---------------- self-reported vs computed experience
    years_source = "computed_from_dates"
    if profile.self_reported_years_experience:
        delta = abs(profile.self_reported_years_experience - computed_years)
        if delta >= 2:
            flags.append(
                f"Self-reported experience ({profile.self_reported_years_experience:g} yrs) disagrees with "
                f"experience computed from dates ({computed_years:g} yrs) by {delta:.1f} yrs"
            )
            years_source = "computed_from_dates (self-report disputed)"

    # ---------------- contact details
    if not profile.email:
        flags.append("No email address found")
    elif not EMAIL_RE.match(profile.email):
        flags.append(f"Malformed email address: {profile.email}")
    if not profile.phone:
        flags.append("No phone number found")

    # ---------------- education sanity
    if not profile.education:
        flags.append("No education section could be extracted")
    tier, highest_degree = "Unknown", None
    for edu in profile.education:
        t = classify_degree(edu.degree)
        if TIER_RANK[t] > TIER_RANK[tier]:
            tier, highest_degree = t, edu.degree
        inst = (edu.institution or "").lower()
        if any(school in inst for school in GRADUATE_ONLY_SCHOOLS) and classify_degree(edu.degree) == "Bachelor's":
            flags.append(
                f"Undergraduate degree attributed to a graduate-only school: {edu.degree} at {edu.institution}"
            )

    seen_edu: set[tuple] = set()
    for edu in profile.education:
        sig = ((edu.institution or "").lower().strip(), (edu.degree or "").lower().strip(), edu.end_year)
        if sig in seen_edu:
            flags.append(f"Duplicate education entry: {edu.degree or '?'} at {edu.institution}")
        seen_edu.add(sig)

    # ---------------- employer contradictions inside one role
    for role in profile.roles:
        employer_l = role.employer.lower()
        blob = " ".join(role.highlights).lower()
        for firm in KNOWN_FIRMS:
            if firm in blob and firm not in employer_l:
                # Only flag when the bullet claims the candidate worked or started there.
                if re.search(rf"\b(at|for|of|with|joined)\s+[^.]{{0,30}}{re.escape(firm)}", blob):
                    flags.append(
                        f"Role at {role.employer} contains a bullet referring to employment at '{firm}' - "
                        f"possible employer mismatch"
                    )
                    break

    # ---------------- coverage / classification completeness
    if not profile.sectors_covered:
        flags.append("No sector coverage could be determined")
    if profile.primary_strategy_type == "Unclear":
        flags.append("Investment strategy style could not be classified")
    if profile.region == "Unknown":
        flags.append("Region could not be determined")

    # Model-reported ambiguities feed the same flag stream, tagged by origin.
    for note in profile.extraction_notes:
        flags.append(f"[model] {note}")
    if profile.extraction_confidence == "low":
        flags.append("Model reported low extraction confidence")

    # De-duplicate while preserving order: a duplicated education entry would otherwise
    # raise the same flag twice and double-penalise the quality score.
    flags = list(dict.fromkeys(flags))

    # ---------------- quality score
    hard_flags = [f for f in flags if not f.startswith("[model]")]
    score = max(0.0, round(1.0 - 0.08 * len(hard_flags) - 0.03 * (len(flags) - len(hard_flags)), 2))

    employers = list(dict.fromkeys(r.employer for r in profile.roles if r.employer))
    coverage_sizes = [r.coverage_universe_size for r in profile.roles if r.coverage_universe_size]

    searchable = " ".join([
        profile.full_name,
        profile.current_employer or "",
        profile.current_title or "",
        " ".join(employers),
        " ".join(r.title or "" for r in profile.roles),
        " ".join(profile.sectors_covered),
        " ".join(profile.sector_specialisation_detail),
        " ".join(profile.geographic_markets_covered),
        " ".join(profile.programming_languages),
        " ".join(profile.tools_and_platforms),
        " ".join(profile.certifications),
        " ".join(e.institution for e in profile.education),
        " ".join(e.degree or "" for e in profile.education),
        " ".join(h for r in profile.roles for h in r.highlights),
    ]).lower()

    base = profile.model_dump()
    base.pop("max_coverage_universe", None)  # recomputed below from role-level evidence
    return EnrichedCandidate(
        **base,
        candidate_id=re.sub(r"[^a-z0-9]+", "-", profile.full_name.lower()).strip("-"),
        computed_years_experience=computed_years,
        computed_years_excluding_internships=computed_years_ex_intern,
        years_experience_source=years_source,
        career_start_year=(min(starts) // 12) if starts else None,
        n_roles=len(profile.roles),
        n_employers=len(employers),
        employer_list=employers,
        highest_degree_tier=tier,
        highest_degree=highest_degree,
        is_currently_employed=is_employed,
        months_since_last_role=months_since_last,
        career_gaps=gaps,
        data_quality_flags=flags,
        data_quality_score=score,
        max_coverage_universe=max(coverage_sizes) if coverage_sizes else profile.max_coverage_universe,
        searchable_text=searchable,
    )


def enrich_all(profiles: list[CandidateProfile]) -> list[EnrichedCandidate]:
    return [enrich(p) for p in profiles]


def corpus_quality_summary(candidates: list[EnrichedCandidate]) -> dict:
    """Aggregate quality view for the notebook and the app's data-quality tab."""
    total_flags = sum(len(c.data_quality_flags) for c in candidates)
    return {
        "candidates": len(candidates),
        "with_flags": sum(1 for c in candidates if c.data_quality_flags),
        "total_flags": total_flags,
        "mean_quality_score": round(sum(c.data_quality_score for c in candidates) / max(len(candidates), 1), 3),
        "high_confidence": sum(1 for c in candidates if c.extraction_confidence == "high"),
        "currently_employed": sum(1 for c in candidates if c.is_currently_employed),
    }
