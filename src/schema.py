"""Pydantic schema for a parsed candidate profile.

Why a strict schema rather than free-form JSON:

* It is handed to the LLM as a JSON Schema, so the model is constrained at generation
  time rather than corrected afterwards.
* It fails loudly. A hallucinated field or a string where a number belongs raises a
  validation error we can retry, instead of quietly poisoning the search index.
* It defines the controlled vocabularies (region, sector, strategy) that make
  filtering possible. Without normalisation, "TMT", "Tech/Media" and "Internet"
  would be three unrelated filter values.

Design decision worth defending in review: the model is asked only for what is
*stated* in the document. Every derived quantity - total years of experience,
seniority tier, career gaps, match scores - is computed in Python from the extracted
dates. LLMs are good at reading and bad at arithmetic, and derived numbers must be
reproducible and auditable.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Region = Literal["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa", "Unknown"]
Sector = Literal[
    "Technology", "Media & Telecom", "Healthcare", "Financial Services", "Energy",
    "Industrials", "Consumer", "Real Estate", "Utilities", "Materials",
    "Credit", "Macro / Rates & FX", "Multi-Sector / Generalist",
]
StrategyType = Literal["Fundamental", "Systematic / Quantitative", "Hybrid", "Unclear"]
FirmType = Literal[
    "Hedge Fund", "Asset Manager", "Investment Bank - Sell-Side Research",
    "Investment Bank - Banking / Markets", "Private Equity / Venture Capital",
    "Commercial / Corporate Bank", "Consulting", "Corporate / Industry",
    "Academic / Research", "Other",
]
MarketSide = Literal["Buy-Side", "Sell-Side", "Private Markets", "Corporate", "Academic", "Unknown"]
Seniority = Literal["Intern", "Analyst", "Senior Analyst", "Associate", "Lead Analyst", "Portfolio Manager", "Other"]
Confidence = Literal["high", "medium", "low"]


class Education(BaseModel):
    institution: str
    degree: Optional[str] = Field(None, description="e.g. 'MBA', 'MSc Financial Technology', 'MBBS'")
    field_of_study: Optional[str] = None
    location: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = Field(None, description="Graduation year if stated")
    gpa: Optional[str] = Field(None, description="Verbatim GPA or percentage, e.g. '3.5/4.0', '68%'")
    honors: Optional[str] = None

    @field_validator("start_year", "end_year")
    @classmethod
    def _plausible_year(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1950 <= v <= 2035):
            return None
        return v


class Role(BaseModel):
    """A single position. Dates stay as strings in YYYY-MM so partial dates survive."""
    employer: str
    title: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = Field(None, description="YYYY-MM, or YYYY if only a year is stated")
    end_date: Optional[str] = Field(None, description="YYYY-MM, YYYY, or 'present'")
    duration_stated: Optional[str] = Field(
        None, description="Verbatim duration when the resume gives tenure instead of dates, e.g. '8 years 10 months'"
    )
    is_current: bool = False
    is_internship: bool = False
    firm_type: FirmType = "Other"
    market_side: MarketSide = "Unknown"
    seniority: Seniority = "Other"
    strategy_type: StrategyType = "Unclear"
    sectors: list[Sector] = Field(default_factory=list)
    coverage_universe_size: Optional[int] = Field(
        None, description="Number of names/stocks covered, if stated (e.g. 'coverage of 32 companies' -> 32)"
    )
    aum_or_portfolio_size: Optional[str] = Field(None, description="Verbatim, e.g. '$4.2bn gross portfolio'")
    highlights: list[str] = Field(default_factory=list, description="Up to 4 short achievement summaries")


class CandidateProfile(BaseModel):
    """Everything extracted from one resume, before Python-side enrichment."""

    # ---- provenance
    source_file: str
    source_agency: Optional[str] = Field(
        None, description="Recruiting agency named in a header/footer/watermark, if any"
    )

    # ---- identity
    full_name: str
    honorific: Optional[str] = Field(None, description="e.g. 'Dr.'")
    email: Optional[str] = None
    phone: Optional[str] = None
    location_city: Optional[str] = None
    location_country: Optional[str] = None
    region: Region = "Unknown"

    # ---- headline positioning
    current_employer: Optional[str] = None
    current_title: Optional[str] = None
    primary_strategy_type: StrategyType = "Unclear"
    primary_market_side: MarketSide = "Unknown"
    primary_firm_type: FirmType = "Other"
    seniority_level: Seniority = "Other"
    sectors_covered: list[Sector] = Field(default_factory=list)
    sector_specialisation_detail: list[str] = Field(
        default_factory=list, description="Verbatim sub-sector detail, e.g. 'Digital Health & AI diagnostics'"
    )
    geographic_markets_covered: list[str] = Field(
        default_factory=list, description="Markets the candidate researches, e.g. 'Greater China', 'India'"
    )

    # ---- claims stated by the candidate (kept separate from computed values)
    self_reported_years_experience: Optional[float] = Field(
        None, description="Only if the resume explicitly states a number of years"
    )
    max_coverage_universe: Optional[int] = None

    # ---- structured history
    roles: list[Role] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    # ---- credentials and skills
    certifications: list[str] = Field(default_factory=list, description="e.g. 'CFA Charterholder', 'Series 7'")
    has_cfa: bool = False
    cfa_status: Optional[str] = Field(None, description="'Charterholder', 'Level III passed', 'Level II candidate'")
    has_medical_degree: bool = False
    licenses: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    tools_and_platforms: list[str] = Field(default_factory=list, description="e.g. 'Bloomberg', 'FactSet'")
    languages_spoken: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)

    # ---- model self-assessment
    extraction_confidence: Confidence = "medium"
    extraction_notes: list[str] = Field(
        default_factory=list, description="Ambiguities, contradictions or gaps the model noticed"
    )

    @field_validator("sectors_covered", "programming_languages", "tools_and_platforms",
                     "languages_spoken", "certifications", mode="after")
    @classmethod
    def _dedupe(cls, v: list) -> list:
        seen, out = set(), []
        for item in v:
            key = str(item).strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item)
        return out


class EnrichedCandidate(CandidateProfile):
    """Profile plus everything Python computes. This is what the app indexes."""
    candidate_id: str = ""
    computed_years_experience: float = 0.0
    computed_years_excluding_internships: float = 0.0
    years_experience_source: str = "computed_from_dates"
    career_start_year: Optional[int] = None
    n_roles: int = 0
    n_employers: int = 0
    employer_list: list[str] = Field(default_factory=list)
    highest_degree_tier: str = "Unknown"
    highest_degree: Optional[str] = None
    is_currently_employed: bool = True
    months_since_last_role: Optional[int] = None
    career_gaps: list[str] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)
    data_quality_score: float = 1.0
    searchable_text: str = ""


def llm_json_schema() -> dict:
    """JSON Schema handed to the LLM for structured-output enforcement."""
    return CandidateProfile.model_json_schema()
