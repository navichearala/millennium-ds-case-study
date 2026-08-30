"""Unit tests for the deterministic parts of the pipeline.

The LLM step is inherently probabilistic, so it is covered by schema enforcement and a
labelled evaluation set rather than by unit tests. Everything downstream of it is
ordinary Python and must be pinned by tests, because these are the rules that produce
every number shown to a recruiter.

Each test corresponds to a real behaviour in the sample corpus, noted in its docstring.

Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from document_loader import _dedupe_row_cells, clean_text
from schema import CandidateProfile, Education, Role
from validation import enrich, merge_intervals, parse_duration_to_months, parse_ym


# ------------------------------------------------------------------ date maths
def test_year_only_start_and_end_resolve_to_january_and_december():
    """'2016 - 2019' must span four calendar years, not one month."""
    assert parse_ym("2016") == (2016, 1)
    assert parse_ym("2019", is_end=True) == (2019, 12)


def test_month_precision_and_present_are_parsed():
    assert parse_ym("2021-06") == (2021, 6)
    assert parse_ym("present") is not None
    assert parse_ym("not a date") is None


def test_overlapping_intervals_are_counted_once():
    """Chen Li holds an internship and a research assistantship concurrently."""
    merged = merge_intervals([(0, 24), (12, 36)])
    assert merged == [(0, 36)]
    assert sum(e - s for s, e in merged) == 36


def test_non_overlapping_intervals_are_preserved():
    assert merge_intervals([(0, 12), (24, 36)]) == [(0, 12), (24, 36)]


@pytest.mark.parametrize("text,expected", [
    ("8 years 10 months", 106),
    ("10 months", 10),
    ("2 months", 2),
    ("8 weeks", 2),
    ("no duration here", None),
])
def test_stated_durations_are_parsed(text, expected):
    """Viktor Sharat's resume states tenure as durations with no dates at all."""
    assert parse_duration_to_months(text) == expected


# ------------------------------------------------------------ document cleaning
def test_merged_table_cells_are_deduplicated():
    """Two resumes repeat every table row up to four times via merged cells."""
    assert _dedupe_row_cells(["A", "A", "A", "B", ""]) == ["A", "B"]


def test_ligature_damage_is_repaired():
    """Omar El-Hassan's PDF loses the 'ti' ligature in every affected word."""
    cleaned, repairs = clean_text("Quan\ufffdta\ufffdve Developer")
    assert cleaned == "Quantitative Developer"
    assert any("U+FFFD" in r for r in repairs)


def test_smart_punctuation_is_normalised():
    cleaned, _ = clean_text("Analyst \u2013 TMT \u2018coverage\u2019")
    assert cleaned == "Analyst - TMT 'coverage'"


# ------------------------------------------------------------ validation rules
def _profile(**overrides) -> CandidateProfile:
    base = dict(
        source_file="test.docx", full_name="Test Candidate",
        email="test@example.com", phone="+1 555 0100",
        region="North America", primary_strategy_type="Fundamental",
        sectors_covered=["Healthcare"], roles=[], education=[],
    )
    base.update(overrides)
    return CandidateProfile(**base)


def test_self_reported_experience_conflict_is_flagged():
    """Priya Nakamura claims 9 years; her dates imply 12.7."""
    profile = _profile(
        self_reported_years_experience=9,
        roles=[Role(employer="Firm A", start_date="2013-10", end_date="present", is_current=True)],
    )
    result = enrich(profile)
    assert result.computed_years_experience > 12
    assert any("disagrees" in f for f in result.data_quality_flags)
    assert "self-report disputed" in result.years_experience_source


def test_overlapping_roles_at_one_employer_are_flagged():
    """Omar El-Hassan's internship overlaps his full-time role at BNP Paribas."""
    profile = _profile(roles=[
        Role(employer="BNP Paribas", start_date="2022-05", end_date="present", is_current=True),
        Role(employer="BNP Paribas", start_date="2022-04", end_date="2022-08", is_internship=True),
    ])
    flags = enrich(profile).data_quality_flags
    assert any("Overlapping roles" in f and "same employer" in f for f in flags)


def test_impossible_dates_are_flagged():
    profile = _profile(roles=[Role(employer="Firm A", start_date="2020-06", end_date="2019-01")])
    assert any("Impossible dates" in f for f in enrich(profile).data_quality_flags)


def test_malformed_email_is_flagged():
    """Marcus Chen-Rodriguez's email has no top-level domain."""
    assert any("Malformed email" in f for f in enrich(_profile(email="rchen@hotmail")).data_quality_flags)


def test_undergraduate_degree_at_graduate_only_school_is_flagged():
    """Ryan Patel and Vikram Shah both attribute a bachelor's to a graduate-only school."""
    profile = _profile(education=[
        Education(institution="Northwestern University - Kellogg School of Management",
                  degree="Bachelor of Science in Business Administration", end_year=2014),
    ])
    assert any("graduate-only school" in f for f in enrich(profile).data_quality_flags)


def test_employer_contradiction_inside_a_role_is_flagged():
    """Marina Silva Costa's Bain role credits McKinsey's case competition."""
    profile = _profile(roles=[Role(
        employer="Bain & Company", start_date="2016", end_date="2019",
        highlights=["Led launch of McKinsey's first case competition in Brazil"],
    )])
    assert any("possible employer mismatch" in f for f in enrich(profile).data_quality_flags)


def test_study_period_gaps_are_annotated_not_treated_as_unexplained():
    """An MBA is an explanation for a gap, not a red flag."""
    profile = _profile(
        roles=[
            Role(employer="Consultancy", start_date="2016", end_date="2019"),
            Role(employer="Asset Manager", start_date="2021-08", end_date="present", is_current=True),
        ],
        education=[Education(institution="MIT Sloan", degree="MBA", start_year=2019, end_year=2021)],
    )
    gaps = enrich(profile).career_gaps
    assert len(gaps) == 1
    assert "overlaps" in gaps[0] and "MBA" in gaps[0]


def test_internship_summers_do_not_create_unemployment_gaps():
    profile = _profile(roles=[
        Role(employer="Fund", start_date="2016-05", end_date="2016-08", is_internship=True),
        Role(employer="Employer", start_date="2017-07", end_date="present", is_current=True),
    ])
    assert enrich(profile).career_gaps == []


def test_missing_current_role_is_reported():
    """Chen Li has no role recorded after September 2023."""
    result = enrich(_profile(roles=[Role(employer="Bank", start_date="2021-06", end_date="2023-09")]))
    assert result.is_currently_employed is False
    assert any("Not currently employed" in g for g in result.career_gaps)


def test_duplicate_flags_are_not_double_counted():
    """A duplicated education entry must not penalise the quality score twice."""
    edu = Education(institution="Kellogg School of Management", degree="Bachelor of Science", end_year=2014)
    single = enrich(_profile(education=[edu]))
    doubled = enrich(_profile(education=[edu, edu]))
    graduate_flags = [f for f in doubled.data_quality_flags if "graduate-only" in f]
    assert len(graduate_flags) == 1
    assert doubled.data_quality_score <= single.data_quality_score


def test_degree_tier_picks_the_highest_qualification():
    profile = _profile(education=[
        Education(institution="School", degree="B.M.S", end_year=2007),
        Education(institution="Medical College", degree="M.B.B.S", end_year=2010),
        Education(institution="Business School", degree="PGDM (Finance)", end_year=2012),
    ])
    assert enrich(profile).highest_degree_tier == "Doctorate / Medical"


def test_internships_are_excluded_from_the_experience_total():
    profile = _profile(roles=[
        Role(employer="Fund", start_date="2019-01", end_date="2019-12", is_internship=True),
        Role(employer="Employer", start_date="2020-01", end_date="2021-12"),
    ])
    result = enrich(profile)
    assert result.computed_years_experience > result.computed_years_excluding_internships


def test_clean_record_scores_near_one():
    profile = _profile(
        roles=[Role(employer="Fund", start_date="2018-01", end_date="present", is_current=True,
                    sectors=["Healthcare"])],
        education=[Education(institution="University", degree="MBA", start_year=2015, end_year=2017)],
    )
    result = enrich(profile)
    assert result.data_quality_score >= 0.9
    assert result.data_quality_flags == []
