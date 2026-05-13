"""Page 3 — Recruitment Report.

The operational deliverable. Answers the headline business question:
"What are the most promising job categories and roles to prioritize?"

Sections:
  3.1 Executive summary strip
  3.2 Promising roles (the headline)
  3.3 Category-level recommendations
  3.4 Salary benchmark report (category × seniority)
  3.5 Recruiter action list (powered by lib/suggestions.py)
  3.6 Exports

Consumes: numberOfVacancies, applications_per_vacancy,
mass_hiring_flag, zero_engagement_flag, salary_suspicious_low,
hard_to_fill_score, demand_intensity_score, average_salary,
days_since_posting, category_1, title_seniority,
postedCompany_name.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.chart_helpers import PALETTE, fmt_int, fmt_sgd
from lib.data_loader import (
    engine_badge, filter_signature, get_filtered_df, load_features_or_stop,
    render_data_source_picker, render_data_status, render_execution_mode_toggle,
)
from lib.filters import filter_summary, render_sidebar_filters
from lib.suggestions import ACTION_LIST_LABELS, SEVERITY_BADGE, suggest_for_df

st.set_page_config(page_title="Recruitment Report · HR Recruiter Insights", layout="wide")

render_data_source_picker()
render_execution_mode_toggle()
df = load_features_or_stop()
filters = render_sidebar_filters(df)
SIG = filter_signature(filters)
df_f = get_filtered_df(SIG)
st.sidebar.markdown("---")
st.sidebar.markdown(filter_summary(filters, df_f, df))
render_data_status()

st.title("Recruitment Report")
st.caption(
    "Operational summary for the current filter set. Use the exports at "
    f"the bottom to take this offline. · Engine: **{engine_badge()}**"
)

if df_f.empty:
    st.warning("No postings match current filters. Reset filters in the sidebar.")
    st.stop()


# ─── Cached aggregations ─────────────────────────────────────────────────
# Heavy work cached by filter signature. Eager CSV generation in the
# download buttons is the biggest win — it used to fire every render.

@st.cache_data(show_spinner=False, max_entries=8)
def _promising(sig: tuple) -> pd.DataFrame:
    """Subset matching the spec 3.2 'promising role' definition."""
    d = get_filtered_df(sig)
    cat_apv_p25 = (
        d.groupby("category_1", observed=True)["applications_per_vacancy"].transform(
            lambda s: s.quantile(0.25)
        )
    )
    mask = (
        (d["numberOfVacancies"] >= 3)
        & (d["applications_per_vacancy"] <= cat_apv_p25)
        & (~d["mass_hiring_flag"])
        & (~d["zero_engagement_flag"])
        & (~d["salary_suspicious_low"])
    )
    return d[mask]


@st.cache_data(show_spinner=False, max_entries=8)
def _exec_summary(sig: tuple) -> dict:
    d = get_filtered_df(sig)
    promising = _promising(sig)

    top_promising_cat = (
        promising["category_1"].mode().iloc[0]
        if not promising.empty and not promising["category_1"].mode().empty
        else "—"
    )

    bench = (
        d.groupby("category_1", observed=True)
        .agg(n=("metadata_jobPostId", "count"),
             undisclosed_share=("salary_undisclosed_flag", "mean"))
        .assign(score=lambda x: x["n"] - 100 * x["undisclosed_share"])
    )
    best_bench = bench["score"].idxmax() if not bench.empty else "—"

    sen = (
        d.groupby("title_seniority", observed=True)["demand_intensity_score"]
        .median().dropna()
    )
    top_demand_seniority = sen.idxmax() if not sen.empty else "—"

    largest_seg = d["category_1"].value_counts()
    largest_segment = (largest_seg.idxmax(), int(largest_seg.iloc[0])) if not largest_seg.empty else ("—", 0)

    return {
        "top_promising_cat": top_promising_cat,
        "promising_n": len(promising),
        "best_bench": best_bench,
        "top_demand_seniority": top_demand_seniority,
        "largest_segment": largest_segment,
    }


@st.cache_data(show_spinner=False, max_entries=8)
def _cat_stats(sig: tuple) -> pd.DataFrame:
    d = get_filtered_df(sig)
    stats = (
        d.groupby("category_1", observed=True)
        .agg(
            n=("metadata_jobPostId", "count"),
            median_salary=("average_salary", "median"),
            p25_salary=("average_salary", lambda s: s.quantile(0.25)),
            p75_salary=("average_salary", lambda s: s.quantile(0.75)),
            median_apv=("applications_per_vacancy", "median"),
            median_duration=("posting_duration_days", "median"),
            median_demand=("demand_intensity_score", "median"),
        )
        .sort_values("n", ascending=False)
        .head(10)
        .reset_index()
    )
    stats["demand_rank"] = stats["median_demand"].rank(ascending=False).astype(int)
    return stats


@st.cache_data(show_spinner=False, max_entries=8)
def _overall_medians(sig: tuple) -> dict:
    d = get_filtered_df(sig)
    return {
        "apv": d["applications_per_vacancy"].median(),
        "salary": d["average_salary"].median(),
        "duration": d["posting_duration_days"].median(),
    }


@st.cache_data(show_spinner=False, max_entries=16)
def _salary_matrix(sig: tuple, yoe_lo: int, yoe_hi: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (medians_masked, counts). YoE band is part of the key."""
    d = get_filtered_df(sig)
    sub = d[d["minimumYearsExperience"].between(yoe_lo, yoe_hi)]
    sen_order = ["C-suite", "Director/Head", "Manager/Lead", "Senior", "Mid/Other", "Junior"]
    sen_cols = [s for s in sen_order if s in sub["title_seniority"].astype(str).unique()]
    counts = (
        sub.dropna(subset=["category_1", "title_seniority"])
        .groupby(["category_1", "title_seniority"], observed=True)
        .size().unstack(fill_value=0).reindex(columns=sen_cols)
    )
    medians = (
        sub.dropna(subset=["category_1", "title_seniority"])
        .groupby(["category_1", "title_seniority"], observed=True)["average_salary"]
        .median().unstack().reindex(columns=counts.columns)
    )
    medians_masked = medians.where(counts >= 10)
    top_cats = counts.sum(axis=1).sort_values(ascending=False).head(20).index
    medians_masked = medians_masked.loc[[c for c in top_cats if c in medians_masked.index]]
    return medians_masked, counts


@st.cache_data(show_spinner=False, max_entries=8)
def _suggestions(sig: tuple, cap: int) -> pd.DataFrame:
    d = get_filtered_df(sig)
    src = d if len(d) <= cap else d.sample(cap, random_state=0)
    return suggest_for_df(src)


@st.cache_data(show_spinner=False, max_entries=8)
def _csv_filtered(sig: tuple) -> bytes:
    return get_filtered_df(sig).to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=8)
def _csv_promising_view(sig: tuple, cols: tuple, rename: tuple) -> bytes:
    p = _promising(sig)
    if p.empty:
        return b""
    view = (
        p.nlargest(30, "numberOfVacancies")[list(cols)]
        .rename(columns=dict(rename))
    )
    return view.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=16)
def _csv_matrix(sig: tuple, yoe_lo: int, yoe_hi: int) -> bytes:
    medians_masked, _ = _salary_matrix(sig, yoe_lo, yoe_hi)
    if medians_masked.dropna(how="all").empty:
        return b""
    return medians_masked.to_csv().encode("utf-8")


promising = _promising(SIG)


# ─── 3.1 Executive summary strip ─────────────────────────────────────────
st.subheader("3.1 Executive summary")

es = _exec_summary(SIG)
seg_name, seg_n = es["largest_segment"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Most promising category", str(es["top_promising_cat"]),
          help=f"{es['promising_n']:,} promising roles match the spec definition.")
c2.metric("Best benchmark target", str(es["best_bench"]),
          help="Largest n and lowest undisclosed-salary share.")
c3.metric("Highest-demand seniority", str(es["top_demand_seniority"]),
          help="By median demand_intensity_score.")
c4.metric("Largest market segment", str(seg_name),
          help=f"{seg_n:,} postings." if seg_n else "")

st.markdown("---")


# ─── 3.2 Promising roles ─────────────────────────────────────────────────
st.subheader("3.2 Promising roles")
st.markdown(
    "**Definition.** `numberOfVacancies ≥ 3` AND `applications_per_vacancy ≤ p25` "
    "within category AND NOT mass-hiring AND NOT zero-engagement AND NOT "
    "suspiciously-low salary."
)

promising_cols = [
    "title", "category_1", "postedCompany_name", "numberOfVacancies",
    "applications_per_vacancy", "average_salary", "days_since_posting",
    "hard_to_fill_score",
]
promising_view = (
    promising.nlargest(30, "numberOfVacancies")[promising_cols]
    .rename(columns={
        "title": "Title", "category_1": "Category",
        "postedCompany_name": "Company", "numberOfVacancies": "Vacancies",
        "applications_per_vacancy": "Apps/vacancy",
        "average_salary": "Avg salary",
        "days_since_posting": "Days since post",
        "hard_to_fill_score": "Hard-to-fill",
    })
)
if promising_view.empty:
    st.info("No roles meet the promising criteria in the current filter.")
else:
    st.dataframe(
        promising_view, width="stretch", hide_index=True,
        column_config={
            "Avg salary": st.column_config.NumberColumn(format="S$%d"),
            "Apps/vacancy": st.column_config.NumberColumn(format="%.2f"),
            "Hard-to-fill": st.column_config.NumberColumn(format="%.2f"),
        },
    )

st.markdown("---")


# ─── 3.3 Category-level recommendations ──────────────────────────────────
st.subheader("3.3 Category-level recommendations")

cat_stats = _cat_stats(SIG)
_o = _overall_medians(SIG)
overall_median_apv = _o["apv"]
overall_median_salary = _o["salary"]
overall_median_duration = _o["duration"]


def _recommendation(row: pd.Series) -> tuple[str, str]:
    """Return (label, advice). Three buckets per spec 3.3."""
    high_vacancy = row["n"] >= cat_stats["n"].median()
    low_apv = row["median_apv"] < overall_median_apv
    short_duration = row["median_duration"] < overall_median_duration
    high_apv = row["median_apv"] > overall_median_apv
    low_salary = row["median_salary"] < overall_median_salary

    if high_vacancy and low_apv and short_duration:
        return "High-priority", "Focus sourcing here; the market is open."
    if high_apv and low_salary:
        return "Saturated", "Plenty of supply; selectivity is key."
    return "Steady state", "Maintain current strategy."


_LABEL_COLOR = {
    "High-priority": PALETTE["critical"],
    "Steady state": PALETTE["secondary"],
    "Saturated": PALETTE["warning"],
}

if cat_stats.empty:
    st.info("Not enough category data for the current filter.")
else:
    rows = [cat_stats.iloc[i] for i in range(len(cat_stats))]
    # Render 2-column grid
    for i in range(0, len(rows), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(rows):
                continue
            row = rows[i + j]
            label, advice = _recommendation(row)
            color = _LABEL_COLOR[label]
            with col, st.container(border=True):
                st.markdown(
                    f"#### {row['category_1']}<br>"
                    f"<span style='color:{color}; font-weight:600'>{label}</span>",
                    unsafe_allow_html=True,
                )
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Median salary", fmt_sgd(row["median_salary"]))
                mc2.metric("Demand rank", f"#{row['demand_rank']}")
                mc3.metric("Median duration",
                           f"{int(row['median_duration'])} d"
                           if pd.notna(row["median_duration"]) else "—")
                # Escape `$` — `st.caption` runs markdown which would
                # otherwise treat S$X – S$Y as a LaTeX math block.
                p25_md = fmt_sgd(row['p25_salary']).replace("$", "\\$")
                p75_md = fmt_sgd(row['p75_salary']).replace("$", "\\$")
                st.caption(
                    f"P25–P75: {p25_md} – {p75_md} · n={int(row['n']):,}"
                )
                st.markdown(f"**Recommendation.** {advice}")

st.markdown("---")


# ─── 3.4 Salary benchmark report ─────────────────────────────────────────
st.subheader("3.4 Salary benchmark (category × seniority)")

yoe_lo, yoe_hi = st.slider(
    "Filter by years of experience for the matrix below",
    min_value=0, max_value=20, value=(0, 20),
    key="bench_yoe",
)
medians_masked, counts = _salary_matrix(SIG, yoe_lo, yoe_hi)

if medians_masked.dropna(how="all").empty:
    st.info("No category × seniority cells have ≥10 postings in this slice.")
else:
    display = medians_masked.map(
        lambda v: f"S${v:,.0f}" if pd.notna(v) else "—"
    )
    st.dataframe(display, width="stretch")
    st.caption("Cells with fewer than 10 postings are shown as '—'.")

st.markdown("---")


# ─── 3.5 Recruiter action list ───────────────────────────────────────────
st.subheader("3.5 Recruiter action list")
st.caption("Per-posting suggestions generated by `lib/suggestions.py`. Expand each line to see the rows.")

# Suggestion engine is cached per filter signature — eager run that
# only refires when filters change. Sample cap keeps cold-cache time low.
SUGGEST_CAP = 50_000
sugg = _suggestions(SIG, SUGGEST_CAP)
if len(df_f) > SUGGEST_CAP:
    st.caption(f"Sampled {SUGGEST_CAP:,} of {len(df_f):,} rows for the action list.")

if sugg.empty:
    st.info("No actions to surface from the current filter set.")
else:
    rule_order = [
        ("compensation_bottom_quintile", "High"),
        ("hard_to_fill", "Medium"),
        ("low_conversion_listing", "Medium"),
        ("reposted_stale", "Medium"),
        ("seniority_mismatch", "Low"),
        ("high_applicant_volume", "Low"),
        ("salary_undisclosed", "Low"),
    ]
    for rule, severity in rule_order:
        rows = sugg[sugg["rule"] == rule]
        if rows.empty:
            continue
        label = ACTION_LIST_LABELS[rule].format(n=len(rows))
        badge = SEVERITY_BADGE[severity]
        with st.expander(f"{badge} **{label}**"):
            preview = rows.head(50)[["category_1", "title", "postedCompany_name", "message"]]
            preview.columns = ["Category", "Title", "Company", "Message"]
            st.dataframe(preview, width="stretch", hide_index=True)
            if len(rows) > 50:
                st.caption(f"Showing top 50 of {len(rows):,}. Use exports below for the full list.")

st.markdown("---")


# ─── 3.6 Exports ─────────────────────────────────────────────────────────
st.subheader("3.6 Exports")

_promising_cols = (
    "title", "category_1", "postedCompany_name", "numberOfVacancies",
    "applications_per_vacancy", "average_salary", "days_since_posting",
    "hard_to_fill_score",
)
_promising_rename = (
    ("title", "Title"), ("category_1", "Category"),
    ("postedCompany_name", "Company"), ("numberOfVacancies", "Vacancies"),
    ("applications_per_vacancy", "Apps/vacancy"),
    ("average_salary", "Avg salary"),
    ("days_since_posting", "Days since post"),
    ("hard_to_fill_score", "Hard-to-fill"),
)

e1, e2, e3 = st.columns(3)

# The full filtered CSV is heavy on 1M rows (~5–10s to encode), so it
# stays opt-in. Click "Prepare" to trigger encoding; the download button
# appears once it's ready.
with e1:
    prepared_for = st.session_state.get("_full_csv_for_sig")
    if prepared_for != SIG:
        if st.button("📦 Prepare filtered dataset (CSV)", width="stretch",
                      help="Encodes the full filtered set (~5–10s on 1M rows). Click only when you want to download."):
            with st.spinner("Encoding CSV…"):
                _csv_filtered(SIG)  # warms the cache
            st.session_state["_full_csv_for_sig"] = SIG
            st.rerun()
    else:
        st.download_button(
            "⬇️ Download filtered dataset (CSV)",
            data=_csv_filtered(SIG),
            file_name="mcf_filtered.csv", mime="text/csv",
            width="stretch",
        )

with e2:
    promising_csv = _csv_promising_view(SIG, _promising_cols, _promising_rename)
    st.download_button(
        "⬇️ Download promising roles (CSV)",
        data=promising_csv,
        file_name="mcf_promising_roles.csv", mime="text/csv",
        disabled=len(promising_csv) == 0, width="stretch",
    )
with e3:
    matrix_csv = _csv_matrix(SIG, yoe_lo, yoe_hi)
    st.download_button(
        "⬇️ Download salary matrix (CSV)",
        data=matrix_csv, file_name="mcf_salary_matrix.csv", mime="text/csv",
        disabled=len(matrix_csv) == 0, width="stretch",
    )
