"""Cached parquet loader and shared filter-application logic.

`load_features()` is the single entry point for the feature frame —
every page calls it. Streamlit caches it once per session so the 70 MB
parquet is only read once.

`apply_filters(df, filters)` consumes the dict produced by
`render_sidebar_filters` and returns a filtered view. Pages should not
filter manually; using this helper keeps filter semantics consistent
across the app.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# solutions/data/mcf_features.pkl relative to this file
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "mcf_features.pkl"


@st.cache_data(show_spinner="Loading MCF feature data…")
def load_features() -> pd.DataFrame:
    """Load the pre-built feature pickle. Cached for the session."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature pickle not found at {DATA_PATH}. "
            "Build it first with: "
            "python -c \"import pandas as pd; from feature_engineering import build_features; "
            "build_features(pd.read_pickle('../data/clean_job_step1.pkl'))"
            ".to_pickle('data/mcf_features.pkl')\""
        )
    return pd.read_pickle(DATA_PATH)


def filter_signature(filters: dict) -> tuple:
    """Hashable, deterministic encoding of the filter dict.

    Used as a cache key for `get_filtered_df`. Lists are converted to
    tuples; dates and primitives pass through. Two equivalent filter
    dicts always produce the same signature.
    """
    def _conv(v):
        if isinstance(v, list):
            return ("L", tuple(v))
        if isinstance(v, tuple):
            return ("T", v)  # already hashable; preserve element types
        return v
    return tuple(sorted((k, _conv(v)) for k, v in filters.items()))


def _from_signature(sig: tuple) -> dict:
    out = {}
    for k, v in sig:
        if isinstance(v, tuple) and len(v) == 2 and v[0] in ("L", "T"):
            out[k] = list(v[1]) if v[0] == "L" else v[1]
        else:
            out[k] = v
    return out


@st.cache_data(show_spinner=False, max_entries=8)
def get_filtered_df(sig: tuple) -> pd.DataFrame:
    """Return the filtered feature frame, cached by filter signature.

    Pages call this instead of `apply_filters(df, filters)` directly.
    When the user clicks an expander or otherwise triggers a re-render
    without changing filters, the result is served from cache — no
    re-scan of the 1M-row frame.
    """
    df = load_features()
    return apply_filters(df, _from_signature(sig))


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the sidebar-filter dict to a feature frame.

    Filter keys (any may be absent — missing == no constraint):
      date_range:               (start, end) tuple of date-like
      categories:               list[str]   on category_1
      seniorities:              list[str]   on title_seniority
      salary_bands:             list[str]   on salary_band
      yoe_range:                (lo, hi)    on minimumYearsExperience
      employment_types:         list[str]   on employmentTypes
      exclude_zero_engagement:  bool        drop zero_engagement_flag rows
      exclude_mass_hiring:      bool        drop mass_hiring_flag rows
      exclude_suspicious_low:   bool        drop salary_suspicious_low rows
      employer_type:            "Both" | "Direct only" | "Agency only"
    """
    mask = pd.Series(True, index=df.index)

    date_range = filters.get("date_range")
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= df["metadata_originalPostingDate"].between(start, end)

    if filters.get("categories"):
        mask &= df["category_1"].isin(filters["categories"])

    if filters.get("seniorities"):
        mask &= df["title_seniority"].astype(str).isin(filters["seniorities"])

    if filters.get("salary_bands"):
        mask &= df["salary_band"].astype(str).isin(filters["salary_bands"])

    yoe_range = filters.get("yoe_range")
    if yoe_range and yoe_range != (0, 20):
        mask &= df["minimumYearsExperience"].between(yoe_range[0], yoe_range[1])

    if filters.get("employment_types"):
        mask &= df["employmentTypes"].astype(str).isin(filters["employment_types"])

    if filters.get("exclude_zero_engagement"):
        mask &= ~df["zero_engagement_flag"]
    if filters.get("exclude_mass_hiring"):
        mask &= ~df["mass_hiring_flag"]
    if filters.get("exclude_suspicious_low"):
        mask &= ~df["salary_suspicious_low"]

    employer = filters.get("employer_type", "Both")
    if employer == "Direct only":
        mask &= ~df["is_agency"]
    elif employer == "Agency only":
        mask &= df["is_agency"]

    return df[mask]
