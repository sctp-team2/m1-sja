"""Page 4 — Tools.

Interactive widgets that compute on demand rather than displaying
pre-built views. Lives separately from Deep Analysis so the latter
stays focused on scrollable insights.

Today: salary benchmark calculator. Future home for any other
recruiter utility (compensation gap calc, equity tier estimator, etc.).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.chart_helpers import PALETTE, fmt_int, fmt_sgd
from lib.data_loader import (
    filter_signature, get_filtered_df, load_features,
    render_data_source_picker,
)
from lib.filters import filter_summary, render_sidebar_filters

st.set_page_config(page_title="Tools · MCF Insights", layout="wide")

render_data_source_picker()
df = load_features()
filters = render_sidebar_filters(df)
SIG = filter_signature(filters)
df_f = get_filtered_df(SIG)
st.sidebar.markdown("---")
st.sidebar.markdown(filter_summary(filters, df_f, df))

st.title("Tools")
st.caption(
    "Interactive recruiter utilities. Sidebar filters constrain the "
    "underlying market data the tools draw from — narrow them to focus "
    "on a segment, or leave wide for an all-market benchmark."
)

if df_f.empty:
    st.warning("No postings match the current filters. Reset filters in the sidebar.")
    st.stop()


# ─── Salary benchmark calculator ─────────────────────────────────────────
st.subheader("💰 Salary benchmark calculator")
st.markdown(
    "Pick a **category × seniority × years-of-experience** segment and "
    "see what the market pays. Returns P25 / median / P75 plus sample "
    "size so you can judge how reliable the number is. Backs the same "
    "math as Page 2's salary-positioning section."
)


@st.cache_data(show_spinner=False, max_entries=64)
def _segment_quantiles(sig: tuple, cat: str, sen: str, yoe: int) -> dict | None:
    d = get_filtered_df(sig)
    mask = (
        (d["category_1"].astype(str) == cat)
        & (d["title_seniority"].astype(str) == sen)
        & (d["minimumYearsExperience"] == yoe)
    )
    seg = d.loc[mask, "average_salary"].dropna()
    if seg.empty:
        return None
    return {
        "n": len(seg),
        "p25": float(seg.quantile(0.25)),
        "median": float(seg.median()),
        "p75": float(seg.quantile(0.75)),
    }


with st.container(border=True):
    with st.form("salary_calc"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cat = st.selectbox(
                "Category",
                sorted(df_f["category_1"].dropna().astype(str).unique()),
            )
        with c2:
            sen_options = [
                s for s in
                ["C-suite", "Director/Head", "Manager/Lead", "Senior", "Mid/Other", "Junior"]
                if s in df_f["title_seniority"].astype(str).unique()
            ]
            sen = st.selectbox("Seniority", sen_options)
        with c3:
            yoe = st.slider("Years of experience", 0, 20, 3)
        submitted = st.form_submit_button("Calculate benchmark", type="primary")

    if submitted:
        res = _segment_quantiles(SIG, cat, sen, yoe)
        if res is None:
            st.info(
                "No postings match this exact segment. Try a different "
                "seniority or widen the YoE — small segments often "
                "have zero exact matches in the filtered slice."
            )
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sample size", fmt_int(res["n"]))
            m2.metric("P25", fmt_sgd(res["p25"]))
            m3.metric("Median", fmt_sgd(res["median"]))
            m4.metric("P75", fmt_sgd(res["p75"]))
            # Pre-compute the escaped strings — Python 3.10 disallows
            # backslashes inside f-string `{}` expressions, and Streamlit
            # markdown otherwise treats `$X$` as LaTeX math.
            median_md = fmt_sgd(res['median']).replace('$', '\\$')
            p25_md = fmt_sgd(res['p25']).replace('$', '\\$')
            p75_md = fmt_sgd(res['p75']).replace('$', '\\$')
            st.markdown(
                f"**Interpretation.** A typical {sen} {cat} role with "
                f"{yoe} YoE pays around **{median_md}** per month. "
                f"Offers under **{p25_md}** sit in the bottom quartile "
                f"of this segment; over **{p75_md}** is top-quartile."
            )


# ─── (Future tools placeholder) ──────────────────────────────────────────
st.markdown("---")
st.caption(
    "More tools can be added here as recruiters request them — "
    "compensation-gap auditor, equity-tier estimator, JD checklist, etc."
)
