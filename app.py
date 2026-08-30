"""Millennium BD - Candidate Resume Search Platform (Streamlit).

Run:
    streamlit run app.py

Design principles
-----------------
* **Requisition-first.** BD does not browse candidates, it fills mandates. The app is
  therefore organised around a requisition (region + strategy + sectors + experience
  band) with preset mandates, and every candidate is scored against it.
* **Transparent scoring.** The match score is a visible weighted sum with a per-
  component breakdown and adjustable weights. A black-box ranking cannot be defended
  to a hiring manager, and an opaque score in a hiring context is a liability.
* **Decision support, not automation.** Nothing is filtered out silently. Data-quality
  flags travel with every candidate, and the source resume text is one click away, so
  a human always verifies before acting.
* **Performance.** Data loads once via `st.cache_data`; filtering runs on a vectorised
  pandas frame rather than per-row Python, so the interaction model holds as the corpus
  grows from 10 to 10,000+ records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402

st.set_page_config(
    page_title="Millennium BD - Candidate Search",
    page_icon="\U0001F50D",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0B3C5D"
ACCENT = "#328CC1"
PALETTE = ["#0B3C5D", "#328CC1", "#1D7874", "#D9B310", "#8C6A5D", "#5B7553", "#A23B72", "#6C757D"]

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}}
      h1, h2, h3 {{color: {PRIMARY}; letter-spacing: -0.01em;}}
      div[data-testid="stMetricValue"] {{font-size: 1.55rem; color: {PRIMARY};}}
      .cand-card {{
        border: 1px solid #E3E7EB; border-left: 4px solid {ACCENT}; border-radius: 8px;
        padding: 1rem 1.15rem; margin-bottom: 0.85rem; background: #FFFFFF;
      }}
      .cand-name {{font-size: 1.06rem; font-weight: 650; color: {PRIMARY};}}
      .cand-sub {{color: #55606B; font-size: 0.88rem; margin-top: 0.1rem;}}
      .pill {{
        display: inline-block; padding: 0.13rem 0.6rem; border-radius: 999px;
        font-size: 0.74rem; margin: 0.12rem 0.22rem 0.12rem 0; background: #EEF3F7; color: {PRIMARY};
      }}
      .pill-warn {{background: #FDF2E3; color: #8A5A00;}}
      .pill-good {{background: #E8F4EC; color: #14622F;}}
      .score {{font-size: 1.45rem; font-weight: 700; color: {ACCENT}; text-align: right;}}
      .muted {{color: #6B7680; font-size: 0.82rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data layer
@st.cache_data(show_spinner=False)
def load_candidates() -> tuple[pd.DataFrame, list[dict]]:
    """Load parsed candidates once per session.

    Cached because every filter change re-runs the whole script top to bottom in
    Streamlit. Without caching, a corpus of any real size would re-read and re-parse
    JSON on each keystroke.
    """
    path = Path(config.CANDIDATES_JSON)
    if not path.exists():
        return pd.DataFrame(), []
    records = json.loads(path.read_text())

    rows = []
    for c in records:
        rows.append({
            "candidate_id": c["candidate_id"],
            "name": (f"{c['honorific']} " if c.get("honorific") else "") + c["full_name"],
            "region": c["region"],
            "location": ", ".join(x for x in [c.get("location_city"), c.get("location_country")] if x) or "Not stated",
            "current_employer": c.get("current_employer") or "No current employer stated",
            "current_title": c.get("current_title") or "No current title stated",
            "employed": bool(c.get("is_currently_employed")),
            "strategy": c["primary_strategy_type"],
            "market_side": c["primary_market_side"],
            "firm_type": c["primary_firm_type"],
            "seniority": c["seniority_level"],
            "sectors": c.get("sectors_covered", []),
            "markets": c.get("geographic_markets_covered", []),
            "years": float(c.get("computed_years_experience") or 0),
            "years_ex_int": float(c.get("computed_years_excluding_internships") or 0),
            "self_reported_years": c.get("self_reported_years_experience"),
            "coverage": c.get("max_coverage_universe"),
            "degree_tier": c.get("highest_degree_tier", "Unknown"),
            "highest_degree": c.get("highest_degree"),
            "has_cfa": bool(c.get("has_cfa")),
            "has_md": bool(c.get("has_medical_degree")),
            "languages": c.get("languages_spoken", []),
            "programming": c.get("programming_languages", []),
            "tools": c.get("tools_and_platforms", []),
            "employers": c.get("employer_list", []),
            "n_employers": c.get("n_employers", 0),
            "quality": float(c.get("data_quality_score") or 0),
            "n_flags": len(c.get("data_quality_flags", [])),
            "confidence": c.get("extraction_confidence", "medium"),
            "agency": c.get("source_agency") or "",
            "source_file": c["source_file"],
            "searchable": c.get("searchable_text", ""),
        })
    return pd.DataFrame(rows), records


@st.cache_data(show_spinner=False)
def load_raw_text(source_file: str) -> str:
    stem = Path(source_file).stem
    path = Path(config.RAW_TEXT_DIR) / f"{stem}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else "Source text not available."


def record_by_id(records: list[dict], cid: str) -> dict:
    return next(c for c in records if c["candidate_id"] == cid)


# --------------------------------------------------------------- match scoring
PRESETS: dict[str, dict] = {
    "-- No requisition (browse all) --": {},
    "US Healthcare Fundamental Analyst (5-10 yrs)": {
        "region": "North America", "strategy": "Fundamental", "sectors": ["Healthcare"],
        "min_years": 5.0, "max_years": 10.0, "seniority": ["Analyst", "Senior Analyst", "Associate"],
    },
    "US TMT Fundamental L/S Analyst (4-12 yrs)": {
        "region": "North America", "strategy": "Fundamental",
        "sectors": ["Technology", "Media & Telecom"], "min_years": 4.0, "max_years": 12.0,
        "seniority": ["Analyst", "Senior Analyst", "Associate"],
    },
    "Europe Systematic / Quant Researcher (2-8 yrs)": {
        "region": "Europe", "strategy": "Systematic / Quantitative",
        "sectors": ["Macro / Rates & FX", "Credit"], "min_years": 2.0, "max_years": 8.0,
    },
    "APAC Healthcare Research Analyst (6-15 yrs)": {
        "region": "Asia-Pacific", "strategy": "Fundamental", "sectors": ["Healthcare"],
        "min_years": 6.0, "max_years": 15.0,
    },
    "Global Credit / Macro Analyst (3-10 yrs)": {
        "strategy": "Any", "sectors": ["Credit", "Macro / Rates & FX"],
        "min_years": 3.0, "max_years": 10.0,
    },
}

DEFAULT_WEIGHTS = {"sector": 35, "region": 20, "strategy": 20, "experience": 15, "credentials": 10}


def score_candidate(row: pd.Series, req: dict, weights: dict[str, int]) -> tuple[float, dict[str, float]]:
    """Transparent weighted match score in [0, 100] plus its component breakdown.

    Every component is a documented ratio, so a recruiter can always be told exactly
    why one candidate outranks another. Nothing here is learned or hidden.
    """
    parts: dict[str, float] = {}

    # Sector: proportion of requested sectors the candidate actually covers. A
    # generalist mandate counts as full coverage.
    wanted = set(req.get("sectors") or [])
    if wanted:
        have = set(row["sectors"])
        overlap = len(wanted & have)
        if "Multi-Sector / Generalist" in have and overlap == 0:
            parts["sector"] = 0.55           # credible but unproven in the target sector
        else:
            parts["sector"] = overlap / len(wanted)
    else:
        parts["sector"] = 1.0

    # Region: exact match, with partial credit for a candidate who researches the
    # target market from elsewhere.
    target_region = req.get("region")
    if target_region and target_region != "Any":
        if row["region"] == target_region:
            parts["region"] = 1.0
        else:
            market_blob = " ".join(row["markets"]).lower()
            hint = {"North America": ["united states", "us", "north america"],
                    "Europe": ["europe", "united kingdom", "emea", "france", "germany"],
                    "Asia-Pacific": ["asia", "china", "india", "japan", "hong kong"],
                    "Latin America": ["latam", "latin america", "brazil"],
                    "Middle East & Africa": ["middle east", "africa", "emea"]}.get(target_region, [])
            parts["region"] = 0.5 if any(h in market_blob for h in hint) else 0.0
    else:
        parts["region"] = 1.0

    # Strategy: hybrid profiles get partial credit for either mandate.
    target_strategy = req.get("strategy")
    if target_strategy and target_strategy != "Any":
        if row["strategy"] == target_strategy:
            parts["strategy"] = 1.0
        elif row["strategy"] == "Hybrid":
            parts["strategy"] = 0.7
        else:
            parts["strategy"] = 0.0
    else:
        parts["strategy"] = 1.0

    # Experience: full credit inside the band, decaying linearly outside it, because a
    # candidate one year outside a band is not a non-match.
    lo, hi = req.get("min_years"), req.get("max_years")
    yrs = row["years"]
    if lo is None and hi is None:
        parts["experience"] = 1.0
    else:
        lo = lo if lo is not None else 0.0
        hi = hi if hi is not None else 60.0
        if lo <= yrs <= hi:
            parts["experience"] = 1.0
        else:
            distance = (lo - yrs) if yrs < lo else (yrs - hi)
            parts["experience"] = max(0.0, 1.0 - distance / 5.0)

    # Credentials: a small bonus pool, never a gate. Seniority alignment is included
    # here because titles are noisy across firms and geographies.
    cred = 0.0
    cred += 0.35 if row["degree_tier"] in {"Master's / MBA", "Doctorate / Medical"} else 0.15
    cred += 0.25 if row["has_cfa"] else 0.0
    cred += 0.15 if (row["has_md"] and "Healthcare" in (req.get("sectors") or [])) else 0.0
    wanted_seniority = req.get("seniority") or []
    cred += 0.25 if (not wanted_seniority or row["seniority"] in wanted_seniority) else 0.0
    parts["credentials"] = min(cred, 1.0)

    total_w = sum(weights.values()) or 1
    score = sum(parts[k] * weights.get(k, 0) for k in parts) / total_w * 100
    contributions = {k: round(parts[k] * weights.get(k, 0) / total_w * 100, 1) for k in parts}
    return round(score, 1), contributions


# ------------------------------------------------------------------- rendering
def pill(text: str, kind: str = "") -> str:
    cls = {"warn": "pill pill-warn", "good": "pill pill-good"}.get(kind, "pill")
    return f'<span class="{cls}">{text}</span>'


def candidate_card(row: pd.Series, contributions: dict[str, float] | None, show_score: bool) -> None:
    pills = [pill(s) for s in row["sectors"][:5]]
    pills.append(pill(row["strategy"], "good" if row["strategy"] != "Unclear" else "warn"))
    pills.append(pill(row["market_side"]))
    if row["has_cfa"]:
        pills.append(pill("CFA", "good"))
    if row["has_md"]:
        pills.append(pill("MD / MBBS", "good"))
    if not row["employed"]:
        pills.append(pill("Not currently employed", "warn"))
    if row["n_flags"]:
        pills.append(pill(f"{row['n_flags']} data flags", "warn"))
    if isinstance(row["agency"], str) and row["agency"].strip():
        pills.append(pill(f"via {row['agency']}", "warn"))

    left, right = st.columns([5, 1])
    with left:
        st.markdown(
            f"""<div class="cand-card">
              <div class="cand-name">{row['name']}</div>
              <div class="cand-sub">{row['current_title']} &middot; {row['current_employer']}</div>
              <div class="cand-sub">{row['location']} &middot; {row['region']} &middot;
                 {row['years']:.1f} yrs experience &middot; {row['seniority']} &middot;
                 {row['degree_tier']}</div>
              <div style="margin-top:0.5rem">{''.join(pills)}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with right:
        if show_score and contributions is not None:
            st.markdown(f'<div class="score">{sum(contributions.values()):.0f}</div>'
                        f'<div class="muted" style="text-align:right">match score</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="score">{row["quality"]:.2f}</div>'
                        f'<div class="muted" style="text-align:right">data quality</div>',
                        unsafe_allow_html=True)


def candidate_detail(rec: dict, contributions: dict[str, float] | None = None) -> None:
    name = (f"{rec['honorific']} " if rec.get("honorific") else "") + rec["full_name"]
    st.markdown(f"### {name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experience (computed)", f"{rec['computed_years_experience']:.1f} yrs")
    c2.metric("Employers", rec.get("n_employers", 0))
    c3.metric("Max coverage universe", rec.get("max_coverage_universe") or "n/a")
    c4.metric("Data quality", f"{rec['data_quality_score']:.2f}")

    if contributions:
        st.markdown("**Why this match score**")
        fig = go.Figure(go.Bar(
            x=list(contributions.values()), y=[k.title() for k in contributions],
            orientation="h", marker_color=ACCENT,
            text=[f"{v:.1f}" for v in contributions.values()], textposition="outside",
        ))
        fig.update_layout(height=210, margin=dict(l=0, r=30, t=6, b=0),
                          xaxis_title="points contributed", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["Career", "Education & skills", "Data quality", "Source resume"])

    with tabs[0]:
        rows = []
        for r in rec["roles"]:
            period = (f"{r['start_date'] or '?'} to {r['end_date'] or ('present' if r['is_current'] else '?')}"
                      if r.get("start_date") else (r.get("duration_stated") or "dates not stated"))
            rows.append({
                "Employer": r["employer"], "Title": r.get("title"), "Period": period,
                "Firm type": r["firm_type"], "Side": r["market_side"], "Style": r["strategy_type"],
                "Sectors": ", ".join(r.get("sectors", [])),
                "Coverage": r.get("coverage_universe_size"),
                "Intern": "yes" if r.get("is_internship") else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for r in rec["roles"]:
            if r.get("highlights"):
                with st.expander(f"{r['employer']} - {r.get('title') or 'role'} highlights"):
                    for h in r["highlights"]:
                        st.markdown(f"- {h}")

    with tabs[1]:
        if rec.get("education"):
            st.dataframe(pd.DataFrame([{
                "Institution": e["institution"], "Degree": e.get("degree"),
                "Field": e.get("field_of_study"), "Completed": e.get("end_year"),
                "Grade": e.get("gpa"), "Honors": e.get("honors"),
            } for e in rec["education"]]), use_container_width=True, hide_index=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Certifications**")
            st.write(", ".join(rec.get("certifications") or []) or "None stated")
            st.markdown("**Programming**")
            st.write(", ".join(rec.get("programming_languages") or []) or "None stated")
            st.markdown("**Languages**")
            st.write(", ".join(rec.get("languages_spoken") or []) or "None stated")
        with cols[1]:
            st.markdown("**Tools and platforms**")
            st.write(", ".join(rec.get("tools_and_platforms") or []) or "None stated")
            st.markdown("**Sector detail (verbatim)**")
            for d in rec.get("sector_specialisation_detail") or []:
                st.markdown(f"- {d}")
            if rec.get("publications"):
                st.markdown("**Publications**")
                for p in rec["publications"]:
                    st.markdown(f"- {p}")

    with tabs[2]:
        st.caption(f"Model extraction confidence: **{rec['extraction_confidence']}**"
                   + (f" &middot; agency-sourced document: **{rec['source_agency']}**" if rec.get("source_agency") else ""))
        rules = [f for f in rec["data_quality_flags"] if not f.startswith("[model]")]
        model_notes = [f.replace("[model] ", "") for f in rec["data_quality_flags"] if f.startswith("[model]")]
        if rec.get("career_gaps"):
            st.markdown("**Career continuity**")
            for g in rec["career_gaps"]:
                st.warning(g, icon="\u26A0\uFE0F")
        if rules:
            st.markdown("**Automated validation findings**")
            for f in rules:
                st.markdown(f"- {f}")
        if model_notes:
            st.markdown("**Ambiguities reported by the extraction model**")
            for f in model_notes:
                st.markdown(f"- {f}")
        if not (rules or model_notes or rec.get("career_gaps")):
            st.success("No data-quality issues detected.")

    with tabs[3]:
        st.caption(f"Extracted text from `{rec['source_file']}` - the exact input the model received.")
        st.text_area("Source resume text", load_raw_text(rec["source_file"]), height=420,
                     label_visibility="collapsed")


# ------------------------------------------------------------------------ main
def main() -> None:
    df, records = load_candidates()
    if df.empty:
        st.error("No parsed candidate data found. Run `python src/pipeline.py --seed-cache` first.")
        st.stop()

    st.title("Candidate Resume Search Platform")
    st.caption("Business Development talent sourcing - parsed resume search, screening and analytics")

    # ---------------- sidebar: requisition + filters
    sb = st.sidebar
    sb.markdown("## Requisition")

    # Requisitions are deep-linkable (?req=<n>) so a recruiter can send a colleague the
    # exact mandate view rather than a description of which filters to set.
    preset_keys = list(PRESETS.keys())
    try:
        default_idx = max(0, min(int(st.query_params.get("req", 0)), len(preset_keys) - 1))
    except (TypeError, ValueError):
        default_idx = 0

    preset_name = sb.selectbox("Preset mandate", preset_keys, index=default_idx)
    preset = PRESETS[preset_name]
    sb.caption(f"Shareable link for this mandate: `?req={preset_keys.index(preset_name)}`")

    # Filter options come from the canonical taxonomies first, with any additional
    # observed values appended. Driving options off the current pool alone would make
    # the interface change shape as data arrives, and would silently drop a filter the
    # moment nobody in the pool happens to match it.
    def options(canonical: list[str], observed) -> list[str]:
        seen = list(canonical)
        for value in sorted(set(observed)):
            if value and value not in seen:
                seen.append(value)
        return seen

    all_sectors = options(config.SECTORS, (s for row in df["sectors"] for s in row))
    all_regions = options(config.REGIONS, df["region"])
    all_strategies = options(config.STRATEGY_TYPES, df["strategy"])
    all_seniority = options(config.SENIORITY_LEVELS, df["seniority"])
    all_firm_types = options(config.FIRM_TYPES, df["firm_type"])
    all_sides = options(["Buy-Side", "Sell-Side", "Private Markets", "Corporate", "Academic"], df["market_side"])
    all_langs = sorted({l.split(" (")[0] for row in df["languages"] for l in row})

    def valid_defaults(values, allowed: list[str]) -> list[str]:
        """Presets must never crash the app if a taxonomy value is absent."""
        return [v for v in (values or []) if v in allowed]

    req_region = sb.selectbox(
        "Target region", ["Any"] + all_regions,
        index=(["Any"] + all_regions).index(preset["region"]) if preset.get("region") in all_regions else 0,
    )
    req_strategy = sb.selectbox(
        "Investment approach", ["Any"] + all_strategies,
        index=(["Any"] + all_strategies).index(preset["strategy"]) if preset.get("strategy") in all_strategies else 0,
    )
    req_sectors = sb.multiselect("Sector coverage", all_sectors,
                                 default=valid_defaults(preset.get("sectors"), all_sectors))
    sector_logic = sb.radio("Sector match", ["Any of these", "All of these"], horizontal=True)

    yr_lo, yr_hi = float(df["years"].min()), float(df["years"].max())
    req_years = sb.slider(
        "Target years of experience", 0.0, max(yr_hi + 2, 20.0),
        (float(preset.get("min_years", 0.0)), float(preset.get("max_years", max(yr_hi, 20.0)))), step=0.5,
    )
    # Experience bands on a requisition are guidance, not a hard boundary. Excluding a
    # candidate for being six months outside a band loses good people, so by default the
    # band drives the score (with linear decay outside it) and only becomes a hard cut if
    # the recruiter explicitly asks for one.
    enforce_years = sb.checkbox("Enforce experience band as a hard filter", value=False)

    with sb.expander("Additional filters", expanded=False):
        st.caption("The mandate's target seniority feeds the match score. Set a filter here only "
                   "if you want to exclude other levels outright.")
        f_seniority = st.multiselect("Seniority (hard filter)", all_seniority, default=[])
        f_firm = st.multiselect("Current firm type", all_firm_types)
        f_side = st.multiselect("Market side", all_sides)
        f_degree = st.multiselect("Highest degree", ["Doctorate / Medical", "Master's / MBA", "Bachelor's", "Unknown"])
        f_lang = st.multiselect("Language spoken", all_langs)
        f_cfa = st.checkbox("CFA charterholder only")
        f_md = st.checkbox("Medical degree only")
        f_employed = st.checkbox("Currently employed only")
        f_quality = st.slider("Minimum data-quality score", 0.0, 1.0, 0.0, 0.05)
        keyword = st.text_input("Keyword search", placeholder="e.g. long/short, backtesting, USFDA, Bloomberg")

    with sb.expander("Scoring weights", expanded=False):
        st.caption("The match score is a visible weighted sum. Adjust to reflect what the mandate really values.")
        weights = {k: st.slider(k.title(), 0, 50, v, 5) for k, v in DEFAULT_WEIGHTS.items()}

    # ---------------- vectorised filtering
    mask = pd.Series(True, index=df.index)
    if req_region != "Any":
        mask &= df["region"] == req_region
    if req_strategy != "Any":
        mask &= df["strategy"].isin([req_strategy, "Hybrid"])
    if req_sectors:
        if sector_logic == "All of these":
            mask &= df["sectors"].apply(lambda s: set(req_sectors).issubset(set(s)))
        else:
            mask &= df["sectors"].apply(lambda s: bool(set(req_sectors) & set(s)))
    if enforce_years:
        mask &= df["years"].between(req_years[0], req_years[1])
    if f_seniority:
        mask &= df["seniority"].isin(f_seniority)
    if f_firm:
        mask &= df["firm_type"].isin(f_firm)
    if f_side:
        mask &= df["market_side"].isin(f_side)
    if f_degree:
        mask &= df["degree_tier"].isin(f_degree)
    if f_lang:
        mask &= df["languages"].apply(lambda ls: any(l.split(" (")[0] in f_lang for l in ls))
    if f_cfa:
        mask &= df["has_cfa"]
    if f_md:
        mask &= df["has_md"]
    if f_employed:
        mask &= df["employed"]
    mask &= df["quality"] >= f_quality
    if keyword.strip():
        terms = [t.strip().lower() for t in keyword.split(",") if t.strip()]
        mask &= df["searchable"].apply(lambda txt: all(t in txt for t in terms))

    filtered = df[mask].copy()

    req = {
        "region": req_region, "strategy": req_strategy, "sectors": req_sectors,
        "min_years": req_years[0], "max_years": req_years[1],
        # Preset seniority informs scoring; an explicit filter selection overrides it.
        "seniority": f_seniority or valid_defaults(preset.get("seniority"), all_seniority),
    }
    # A match score is only meaningful once the requisition constrains something. With
    # every dimension set to "Any", every candidate scores near-perfectly, which is a
    # misleading ranking - so in browse mode we rank by experience and surface data
    # quality instead of a vacuous score.
    scoring_on = bool(req_sectors) or req_region != "Any" or req_strategy != "Any"

    if not filtered.empty:
        scored = filtered.apply(lambda r: score_candidate(r, req, weights), axis=1)
        filtered["match_score"] = [s for s, _ in scored]
        filtered["contributions"] = [c for _, c in scored]
        filtered = (filtered.sort_values(["match_score", "years"], ascending=[False, False])
                    if scoring_on else filtered.sort_values("years", ascending=False))

    # ---------------- header metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Candidates in pool", len(df))
    m2.metric("Matching filters", len(filtered))
    m3.metric("Mean experience", f"{filtered['years'].mean():.1f} yrs" if len(filtered) else "-")
    m4.metric("Top match", f"{filtered['match_score'].max():.0f}" if (len(filtered) and scoring_on) else "-")
    m5.metric("Records with flags", int((df["n_flags"] > 0).sum()))

    if preset_name != "-- No requisition (browse all) --":
        st.info(f"Scoring against preset mandate: **{preset_name}**", icon="\U0001F3AF")

    tab_results, tab_compare, tab_insights, tab_quality = st.tabs(
        ["Search results", "Compare candidates", "Talent pool insights", "Data quality"]
    )

    # ---------------- results
    with tab_results:
        if filtered.empty:
            st.warning("No candidates match these criteria. Try widening the experience band, "
                       "switching sector match to 'Any of these', or clearing the keyword search.")
        else:
            view = st.radio("View", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")
            if view == "Table":
                cols = (["name", "match_score"] if scoring_on else ["name"]) + [
                    "region", "location", "current_employer", "current_title",
                    "strategy", "market_side", "seniority", "years", "coverage", "degree_tier",
                    "has_cfa", "quality", "n_flags",
                ]
                table = filtered[cols].rename(columns={
                    "name": "Candidate", "match_score": "Match", "region": "Region", "location": "Location",
                    "current_employer": "Current firm", "current_title": "Title", "strategy": "Approach",
                    "market_side": "Side", "seniority": "Seniority", "years": "Yrs",
                    "coverage": "Coverage", "degree_tier": "Degree", "has_cfa": "CFA",
                    "quality": "Quality", "n_flags": "Flags",
                })
                st.dataframe(table, use_container_width=True, hide_index=True)
            else:
                if not scoring_on:
                    st.caption("No requisition criteria set, so candidates are ranked by experience and "
                               "the right-hand figure shows data-quality score. Choose a preset mandate or "
                               "set a region, approach or sector to rank by match score.")
                for _, row in filtered.iterrows():
                    candidate_card(row, row["contributions"], scoring_on)

            st.download_button(
                "Download this shortlist (CSV)",
                filtered.drop(columns=["contributions", "searchable"]).to_csv(index=False).encode(),
                file_name="shortlist.csv", mime="text/csv",
            )

            st.divider()
            st.markdown("#### Candidate detail")
            names = filtered["name"].tolist()
            chosen = st.selectbox("Select a candidate to review", names, label_visibility="collapsed")
            row = filtered[filtered["name"] == chosen].iloc[0]
            candidate_detail(record_by_id(records, row["candidate_id"]),
                             row["contributions"] if scoring_on else None)

    # ---------------- comparison
    with tab_compare:
        st.markdown("#### Side-by-side comparison")
        st.caption("Compare shortlisted candidates on the dimensions a hiring manager asks about.")
        # Comparison is drawn from the whole pool, pre-populated with the current
        # shortlist. A recruiter frequently wants to benchmark a shortlisted candidate
        # against someone the filters excluded.
        shortlist = filtered["name"].tolist() if not filtered.empty else []
        picks = st.multiselect("Candidates", df["name"].tolist(),
                               default=shortlist[:3], max_selections=4)
        if len(picks) < 2:
            st.info("Select at least two candidates to compare.")
        else:
            src = filtered if not filtered.empty else df
            sub = src[src["name"].isin(picks)]
            missing = [p for p in picks if p not in sub["name"].tolist()]
            if missing:
                extra = df[df["name"].isin(missing)].copy()
                extra["match_score"] = float("nan")
                sub = pd.concat([sub, extra], ignore_index=True)
            comp = pd.DataFrame({
                r["name"]: {
                    "Match score": (f"{r['match_score']:.1f}" if pd.notna(r.get("match_score")) else "not scored"),
                    "Region": r["region"],
                    "Location": r["location"],
                    "Current firm": r["current_employer"],
                    "Title": r["current_title"],
                    "Approach": r["strategy"],
                    "Market side": r["market_side"],
                    "Firm type": r["firm_type"],
                    "Seniority": r["seniority"],
                    "Years experience": r["years"],
                    "Sectors": ", ".join(r["sectors"]),
                    "Max coverage universe": r["coverage"] or "n/a",
                    "Highest degree": r["highest_degree"] or r["degree_tier"],
                    "CFA": "Yes" if r["has_cfa"] else "No",
                    "Languages": ", ".join(r["languages"]) or "Not stated",
                    "Employers": r["n_employers"],
                    "Data quality": r["quality"],
                    "Flags": r["n_flags"],
                } for _, r in sub.iterrows()
            })
            st.dataframe(comp, use_container_width=True)

            radar_axes = ["Experience", "Sector breadth", "Coverage scale", "Credentials", "Data quality"]
            fig = go.Figure()
            for i, (_, r) in enumerate(sub.iterrows()):
                vals = [
                    min(r["years"] / 15, 1) * 100,
                    min(len(r["sectors"]) / 5, 1) * 100,
                    min((r["coverage"] or 0) / 75, 1) * 100,
                    (35 if r["degree_tier"] in {"Master's / MBA", "Doctorate / Medical"} else 15)
                    + (35 if r["has_cfa"] else 0) + (30 if r["has_md"] else 0),
                    r["quality"] * 100,
                ]
                fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_axes + [radar_axes[0]],
                                              fill="toself", name=r["name"],
                                              line_color=PALETTE[i % len(PALETTE)], opacity=0.65))
            fig.update_layout(height=460, polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                              margin=dict(t=30, b=10), legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Axes are normalised for comparability: experience against a 15-year scale, "
                       "sector breadth against 5 sectors, coverage against a 75-name universe.")

    # ---------------- insights
    with tab_insights:
        st.markdown("#### Talent pool composition")
        scope = st.radio("Scope", ["Full pool", "Filtered results"], horizontal=True)
        data = filtered if (scope == "Filtered results" and not filtered.empty) else df

        c1, c2 = st.columns(2)
        with c1:
            exploded = data.explode("sectors").dropna(subset=["sectors"])
            if not exploded.empty:
                pivot = (exploded.pivot_table(index="sectors", columns="region", values="candidate_id",
                                              aggfunc="count").fillna(0))
                fig = px.imshow(pivot, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                                labels=dict(color="candidates"))
                fig.update_layout(title="Sector coverage by region", height=430,
                                  margin=dict(t=50, l=0, r=0, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            counts = data["strategy"].value_counts().reset_index()
            counts.columns = ["strategy", "n"]
            fig = px.pie(counts, names="strategy", values="n", hole=0.55,
                         color_discrete_sequence=PALETTE)
            fig.update_layout(title="Fundamental vs systematic mix", height=430, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(data, x="years", nbins=10, color_discrete_sequence=[ACCENT])
            fig.update_layout(title="Experience distribution", xaxis_title="years of experience",
                              yaxis_title="candidates", height=380, margin=dict(t=50), bargap=0.08)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fc = data["firm_type"].value_counts().reset_index()
            fc.columns = ["firm_type", "n"]
            fig = px.bar(fc, x="n", y="firm_type", orientation="h", color_discrete_sequence=[PRIMARY])
            fig.update_layout(title="Current firm type", xaxis_title="candidates", yaxis_title="",
                              height=380, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            scat = data.dropna(subset=["coverage"])
            if not scat.empty:
                fig = px.scatter(scat, x="years", y="coverage", color="region", hover_name="name",
                                 size=[14] * len(scat), color_discrete_sequence=PALETTE)
                fig.update_layout(title="Coverage universe vs experience", height=380,
                                  xaxis_title="years of experience", yaxis_title="names covered",
                                  margin=dict(t=50))
                st.plotly_chart(fig, use_container_width=True)
        with c6:
            skills = (data.explode("tools").dropna(subset=["tools"])["tools"]
                      .value_counts().head(12).reset_index())
            skills.columns = ["tool", "n"]
            if not skills.empty:
                fig = px.bar(skills, x="n", y="tool", orientation="h", color_discrete_sequence=[ACCENT])
                fig.update_layout(title="Most common tools and platforms", xaxis_title="candidates",
                                  yaxis_title="", height=380, margin=dict(t=50))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Coverage gaps against the current requisition")
        if req_sectors:
            gap_rows = []
            for sector in req_sectors:
                in_region = df[(df["region"] == req_region) if req_region != "Any" else pd.Series(True, index=df.index)]
                gap_rows.append({
                    "Sector": sector,
                    "In pool": int(df["sectors"].apply(lambda s: sector in s).sum()),
                    "In target region": int(in_region["sectors"].apply(lambda s: sector in s).sum()),
                    "Matching full requisition": int(filtered["sectors"].apply(lambda s: sector in s).sum())
                    if not filtered.empty else 0,
                })
            st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)
            st.caption("Where 'matching full requisition' is zero or low, the pipeline is thin for that "
                       "sector in that market and sourcing effort should be redirected.")
        else:
            st.caption("Select one or more sectors in the requisition to see coverage gaps.")

    # ---------------- data quality
    with tab_quality:
        st.markdown("#### Extraction and data-quality review")
        st.caption("Every flag is raised by a documented rule or reported by the extraction model. "
                   "Nothing is auto-corrected, so a reviewer always sees the original discrepancy.")

        q1, q2, q3 = st.columns(3)
        q1.metric("Mean quality score", f"{df['quality'].mean():.2f}")
        q2.metric("Records with flags", int((df["n_flags"] > 0).sum()))
        q3.metric("Total flags", int(df["n_flags"].sum()))

        flag_rows = []
        for rec in records:
            for f in rec["data_quality_flags"]:
                flag_rows.append({
                    "Candidate": rec["full_name"], "Source": rec["source_file"],
                    "Origin": "extraction model" if f.startswith("[model]") else "validation rule",
                    "Finding": f.replace("[model] ", ""),
                })
            for g in rec.get("career_gaps", []):
                flag_rows.append({"Candidate": rec["full_name"], "Source": rec["source_file"],
                                  "Origin": "validation rule", "Finding": f"Career continuity: {g}"})
        fdf = pd.DataFrame(flag_rows)

        origin = st.multiselect("Filter by origin", sorted(fdf["Origin"].unique()),
                                default=sorted(fdf["Origin"].unique()))
        who = st.multiselect("Filter by candidate", sorted(fdf["Candidate"].unique()))
        shown = fdf[fdf["Origin"].isin(origin)]
        if who:
            shown = shown[shown["Candidate"].isin(who)]
        st.dataframe(shown, use_container_width=True, hide_index=True, height=420)

        fig = px.bar(df.sort_values("quality"), x="quality", y="name", orientation="h",
                     color="quality", color_continuous_scale="RdYlGn", range_color=[0, 1])
        fig.update_layout(title="Data-quality score by candidate", xaxis_title="quality score (1.0 = clean)",
                          yaxis_title="", height=430, margin=dict(t=50), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download data-quality report (CSV)", fdf.to_csv(index=False).encode(),
                           file_name="data_quality_report.csv", mime="text/csv")


if __name__ == "__main__":
    main()
