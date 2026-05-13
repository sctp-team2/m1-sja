"""Sidebar filter widget shared across all pages.

`render_sidebar_filters(df)` draws the filter controls in
`st.sidebar` and returns a dict consumed by
`data_loader.apply_filters`. State persists across pages via
`st.session_state`, so users don't lose context when navigating.

Filter inventory (matches spec section 4):
  1. Date range
  2. Category (category_1)
  3. Seniority (title_seniority)
  4. Salary band
  5. Years of experience
  6. Employment type
  7. Quality filters (expander): zero-engagement, mass-hiring,
     suspicious-low salaries, employer type
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


_FILTER_KEYS = (
    "date_range",
    "categories",
    "seniorities",
    "salary_bands",
    "yoe_range",
    "employment_types",
    "exclude_zero_engagement",
    "exclude_mass_hiring",
    "exclude_suspicious_low",
    "employer_type",
)


def _init_defaults(df: pd.DataFrame) -> None:
    """Seed any missing session_state filter values.

    Idempotent — `setdefault` only writes when the key is absent. Safe to
    call on every script run; needed every run because Streamlit may not
    re-populate widget-bound keys until the widget is rendered, and we
    want defaults available for downstream code regardless.
    """
    min_date = df["metadata_originalPostingDate"].min().date()
    max_date = df["metadata_originalPostingDate"].max().date()
    st.session_state.setdefault("date_range", (min_date, max_date))
    st.session_state.setdefault("categories", [])
    st.session_state.setdefault("seniorities", [])
    st.session_state.setdefault("salary_bands", [])
    st.session_state.setdefault("yoe_range", (0, 20))
    st.session_state.setdefault("employment_types", [])
    st.session_state.setdefault("exclude_zero_engagement", True)
    st.session_state.setdefault("exclude_mass_hiring", True)
    st.session_state.setdefault("exclude_suspicious_low", False)
    st.session_state.setdefault("employer_type", "Both")


def _reset_filters(df: pd.DataFrame) -> None:
    """Reassign filter session_state to defaults.

    Must run inside an `on_click` callback (pre-rerun), not after the
    widgets have been instantiated on the current run. Popping
    widget-bound keys mid-run desyncs Streamlit's internal
    `$$WIDGET_ID-…` mapping and raises a KeyError on the next access.
    """
    min_date = df["metadata_originalPostingDate"].min().date()
    max_date = df["metadata_originalPostingDate"].max().date()
    st.session_state["date_range"] = (min_date, max_date)
    st.session_state["categories"] = []
    st.session_state["seniorities"] = []
    st.session_state["salary_bands"] = []
    st.session_state["yoe_range"] = (0, 20)
    st.session_state["employment_types"] = []
    st.session_state["exclude_zero_engagement"] = True
    st.session_state["exclude_mass_hiring"] = True
    st.session_state["exclude_suspicious_low"] = False
    st.session_state["employer_type"] = "Both"


def render_sidebar_filters(df: pd.DataFrame) -> dict:
    """Render the sidebar and return the active filter dict."""
    _init_defaults(df)

    sb = st.sidebar
    sb.markdown("### Filters")

    min_date = df["metadata_originalPostingDate"].min().date()
    max_date = df["metadata_originalPostingDate"].max().date()

    sb.date_input(
        "Posting date range",
        min_value=min_date,
        max_value=max_date,
        key="date_range",
    )

    sb.multiselect(
        "Category",
        options=sorted(df["category_1"].dropna().astype(str).unique()),
        key="categories",
    )

    seniority_order = ["C-suite", "Director/Head", "Manager/Lead", "Senior", "Mid/Other", "Junior"]
    seniority_opts = [s for s in seniority_order if s in df["title_seniority"].astype(str).unique()]
    sb.multiselect("Seniority", options=seniority_opts, key="seniorities")

    band_order = ["<3k", "3-5k", "5-8k", "8-12k", "12-20k", "20k+"]
    band_opts = [b for b in band_order if b in df["salary_band"].astype(str).unique()]
    sb.multiselect("Salary band", options=band_opts, key="salary_bands")

    sb.slider(
        "Years of experience",
        min_value=0, max_value=20,
        key="yoe_range",
    )

    sb.multiselect(
        "Employment type",
        options=sorted(df["employmentTypes"].dropna().astype(str).unique()),
        key="employment_types",
    )

    with sb.expander("Data quality", expanded=False):
        st.checkbox("Exclude zero-engagement postings", key="exclude_zero_engagement")
        st.checkbox("Exclude mass-hiring postings", key="exclude_mass_hiring")
        st.checkbox("Exclude suspicious-low salaries", key="exclude_suspicious_low")
        st.radio(
            "Employer type",
            options=["Both", "Direct only", "Agency only"],
            key="employer_type",
            horizontal=True,
        )

    sb.divider()
    sb.button(
        "Reset filters",
        width="stretch",
        on_click=_reset_filters,
        args=(df,),
    )

    return {k: st.session_state[k] for k in _FILTER_KEYS}


def filter_summary(filters: dict, df_filtered: pd.DataFrame, df_total: pd.DataFrame) -> str:
    """One-line summary chip describing the active filter set."""
    chips = []
    if filters["categories"]:
        chips.append(f"{len(filters['categories'])} categories")
    if filters["seniorities"]:
        chips.append(f"{len(filters['seniorities'])} seniorities")
    if filters["salary_bands"]:
        chips.append(f"{len(filters['salary_bands'])} salary bands")
    if filters["yoe_range"] != (0, 20):
        chips.append(f"{filters['yoe_range'][0]}–{filters['yoe_range'][1]} YoE")
    if filters["employer_type"] != "Both":
        chips.append(filters["employer_type"])
    chip_text = " · ".join(chips) if chips else "all postings"
    n, total = len(df_filtered), len(df_total)
    return f"**{n:,}** of {total:,} postings ({n / total:.1%}) — {chip_text}"
