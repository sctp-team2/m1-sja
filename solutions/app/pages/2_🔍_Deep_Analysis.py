"""Page 2 — Deep Analysis.

Surface non-obvious signals: where demand outstrips supply, which roles
are hardest to fill, salary positioning vs peers, engagement funnel
underperformers, hidden gems, and agency vs direct splits.

Every section follows Problem → Solution → Chart → Insight → Action.
Insight and Action strings are derived from the filtered df so they
update as the user filters.

Consumes (core): demand_intensity_score, hard_to_fill_score,
salary_zscore_within_category, salary_percentile_within_category,
applications_per_view, apps_per_view_vs_category_median,
hidden_gem_flag, is_agency, posting_duration_days,
metadata_repostCount, metadata_totalNumberOfView,
metadata_totalNumberJobApplication.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.chart_helpers import (
    AGENCY_COLORS, DIVERGING_SCALE, PALETTE, empty_chart, fmt_int,
    fmt_sgd, themed,
)
from lib.data_loader import (
    filter_signature, get_filtered_df, load_features,
)
from lib.filters import filter_summary, render_sidebar_filters

st.set_page_config(page_title="Deep Analysis · MCF Insights", layout="wide")

df = load_features()
filters = render_sidebar_filters(df)
SIG = filter_signature(filters)
df_f = get_filtered_df(SIG)
st.sidebar.markdown("---")
st.sidebar.markdown(filter_summary(filters, df_f, df))


# ─── Cached aggregations ─────────────────────────────────────────────────
# Every section's heavy compute lives here, keyed by filter signature.
# Switching filters re-fires once; clicking expanders is free.

@st.cache_data(show_spinner=False, max_entries=8)
def _demand_pivot(sig: tuple) -> pd.DataFrame:
    d = get_filtered_df(sig)
    pivot = (
        d.dropna(subset=["category_1", "title_seniority"])
        .groupby(["category_1", "title_seniority"], observed=True)["demand_intensity_score"]
        .median().unstack()
    )
    cat_n = d["category_1"].value_counts().head(20).index
    pivot = pivot.loc[[c for c in cat_n if c in pivot.index]]
    sen_order = ["C-suite", "Director/Head", "Manager/Lead", "Senior", "Mid/Other", "Junior"]
    return pivot.reindex(columns=[s for s in sen_order if s in pivot.columns])


@st.cache_data(show_spinner=False, max_entries=8)
def _demand_top_cell(sig: tuple) -> tuple | None:
    d = get_filtered_df(sig)
    g = (
        d.dropna(subset=["category_1", "title_seniority"])
        .groupby(["category_1", "title_seniority"], observed=True)["demand_intensity_score"]
        .median()
    )
    if g.empty:
        return None
    top = g.idxmax()
    median_overall = d["applications_per_vacancy"].median()
    median_cell = (
        d.loc[(d["category_1"] == top[0]) & (d["title_seniority"].astype(str) == top[1]),
              "applications_per_vacancy"].median()
    )
    return (top[0], top[1], median_cell, median_overall)


@st.cache_data(show_spinner=False, max_entries=8)
def _hard_to_fill_table_cached(sig: tuple) -> pd.DataFrame:
    d = get_filtered_df(sig)
    cols = ["title", "category_1", "postedCompany_name", "posting_duration_days",
            "metadata_repostCount", "applications_per_vacancy", "hard_to_fill_score"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    return (
        d.dropna(subset=["hard_to_fill_score"])
        .nlargest(20, "hard_to_fill_score")
        .loc[:, cols]
        .rename(columns={
            "title": "Title", "category_1": "Category",
            "postedCompany_name": "Company",
            "posting_duration_days": "Duration (days)",
            "metadata_repostCount": "Reposts",
            "applications_per_vacancy": "Apps/vacancy",
            "hard_to_fill_score": "Hard-to-fill",
        })
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _salary_top_cats(sig: tuple) -> pd.DataFrame:
    d = get_filtered_df(sig)
    top_cats = d["category_1"].value_counts().head(12).index
    sub = d[d["category_1"].isin(top_cats)].copy()
    sub["average_salary"] = sub["average_salary"].clip(upper=25_000)
    return sub[["category_1", "average_salary"]]


@st.cache_data(show_spinner=False, max_entries=8)
def _salary_spread_top(sig: tuple) -> tuple | None:
    d = get_filtered_df(sig)
    top_cats = d["category_1"].value_counts().head(12).index
    sub = d[d["category_1"].isin(top_cats)]
    if sub.empty:
        return None
    q = sub.groupby("category_1", observed=True)["average_salary"].quantile([0.25, 0.75]).unstack()
    spread = q[0.75] - q[0.25]
    cat = spread.idxmax()
    return (cat, q.loc[cat, 0.25], q.loc[cat, 0.75])


@st.cache_data(show_spinner=False, max_entries=8)
def _scatter_sample(sig: tuple) -> pd.DataFrame:
    d = get_filtered_df(sig)
    points = d.dropna(subset=["metadata_totalNumberOfView",
                              "metadata_totalNumberJobApplication",
                              "apps_per_view_vs_category_median"])
    if len(points) > 5000:
        points = points.sample(5000, random_state=0)
    return points[["metadata_totalNumberOfView",
                   "metadata_totalNumberJobApplication",
                   "apps_per_view_vs_category_median"]]


@st.cache_data(show_spinner=False, max_entries=8)
def _scatter_underperformer_count(sig: tuple) -> int | None:
    d = get_filtered_df(sig)
    sub = d.dropna(subset=["metadata_totalNumberOfView",
                            "apps_per_view_vs_category_median"])
    if sub.empty:
        return None
    view_med = sub["metadata_totalNumberOfView"].median()
    return int(((sub["metadata_totalNumberOfView"] > view_med)
                & (sub["apps_per_view_vs_category_median"] < 0)).sum())


@st.cache_data(show_spinner=False, max_entries=8)
def _hidden_gems_table_cached(sig: tuple) -> pd.DataFrame:
    d = get_filtered_df(sig)
    cols = ["title", "category_1", "postedCompany_name",
            "metadata_totalNumberOfView", "metadata_totalNumberJobApplication",
            "applications_per_view"]
    sub = d[d["hidden_gem_flag"]]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    return (
        sub.nlargest(20, "applications_per_view")
        .loc[:, cols]
        .rename(columns={
            "title": "Title", "category_1": "Category",
            "postedCompany_name": "Company",
            "metadata_totalNumberOfView": "Views",
            "metadata_totalNumberJobApplication": "Applications",
            "applications_per_view": "Conv. rate",
        })
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _hidden_gems_summary(sig: tuple) -> tuple | None:
    d = get_filtered_df(sig)
    sub = d[d["hidden_gem_flag"]]
    if sub.empty:
        return None
    top = sub["category_1"].mode().iloc[0] if not sub["category_1"].mode().empty else "—"
    return (len(sub), top)


@st.cache_data(show_spinner=False, max_entries=8)
def _agency_vs_direct_stats(sig: tuple) -> dict:
    d = get_filtered_df(sig)
    return {
        "agency_share": d["is_agency"].mean(),
        "median_direct": d.loc[~d["is_agency"], "average_salary"].median(),
        "median_agency": d.loc[d["is_agency"], "average_salary"].median(),
    }


@st.cache_data(show_spinner=False, max_entries=8)
def _agency_vs_direct_sample(sig: tuple) -> pd.DataFrame:
    """30k stratified sample for the 2×2 facet box plots — full
    1M rows make Plotly box rendering laggy."""
    d = get_filtered_df(sig)
    if len(d) <= 30_000:
        return d[["is_agency", "average_salary", "applications_per_view",
                  "posting_duration_days", "title_seniority"]].copy()
    return (
        d.groupby("is_agency", observed=True, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 15_000), random_state=0))
        [["is_agency", "average_salary", "applications_per_view",
          "posting_duration_days", "title_seniority"]]
        .copy()
    )

st.title("Deep Analysis")
st.caption(
    "Problem → Solution → Chart → Insight → Action. Insight and Action "
    "lines update with your filters."
)

if df_f.empty:
    st.warning("No postings match current filters. Reset filters in the sidebar.")
    st.stop()


# ─── Section template ────────────────────────────────────────────────────
def render_analysis_section(
    *,
    title: str,
    problem: str,
    solution: str,
    chart_fn: Callable[[tuple], None],
    insight_fn: Callable[[tuple], str],
    action_fn: Callable[[tuple], str],
    sig: tuple,
) -> None:
    """Render one Problem → Solution → Chart → Insight → Action section.

    Chart, insight and action functions take the filter signature `sig`
    and look up their pre-aggregated data from the cached helpers —
    nothing is recomputed on expander clicks.
    """
    st.subheader(title)
    with st.container(border=True):
        with st.expander("💡 Why this section? (problem + approach)", expanded=False):
            st.markdown(f"**The problem.** {problem}")
            st.markdown(f"**Our approach.** {solution}")
        chart_fn(sig)
        st.markdown(f"**🔎 Key insight.** {insight_fn(sig)}")
        st.markdown(f"**✅ Recommended action.** {action_fn(sig)}")
    st.markdown("")


# ─── 2.1 Where is demand outstripping supply? ────────────────────────────
def _chart_21(sig: tuple) -> None:
    pivot = _demand_pivot(sig)
    if pivot.empty:
        st.plotly_chart(empty_chart(), width='stretch')
        return

    fig = px.imshow(
        pivot.values,
        x=pivot.columns.astype(str), y=pivot.index.astype(str),
        color_continuous_scale=DIVERGING_SCALE, zmin=-2, zmax=2,
        labels=dict(x="Seniority", y="Category", color="Median score"),
        aspect="auto", text_auto=".2f",
    )
    # Cell numbers in black for contrast against the diverging palette;
    # bigger font so values are readable at a glance.
    fig.update_traces(
        textfont=dict(size=13, color=PALETTE["text"]),
        hovertemplate="%{y} · %{x}<br>median score: %{z:.2f}<extra></extra>",
    )
    fig.update_layout(
        height=560, margin=dict(l=200, r=20, t=20, b=80),
        coloraxis_colorbar=dict(
            title=dict(text="Median<br>demand", font=dict(size=12)),
            tickfont=dict(size=11),
        ),
    )
    st.plotly_chart(themed(fig), width='stretch')
    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "Each cell is the **median `demand_intensity_score`** for that "
            "category × seniority pair. **Red cells = competitive** (many "
            "applicants per vacancy — recruiters can be selective). **Blue "
            "cells = hard to attract** (few applicants — broaden requirements "
            "or sweeten the offer). The score is a z-score, so 0 ≈ overall "
            "median, |1| ≈ one standard deviation away."
        )


def _insight_21(sig: tuple) -> str:
    res = _demand_top_cell(sig)
    if res is None:
        return "Not enough data to identify the most competitive segment."
    cat, sen, median_cell, median_overall = res
    if median_overall and median_cell and median_overall > 0:
        ratio = median_cell / median_overall
        return (f"Most competitive segment is **{cat} · {sen}** — "
                f"~{ratio:.1f}× median applicants per vacancy.")
    return f"Most competitive segment is **{cat} · {sen}**."


def _action_21(sig: tuple) -> str:
    return ("Raise the hiring bar in red cells (competition is high); broaden "
            "requirements or sweeten offers in blue cells (low applicant supply).")


render_analysis_section(
    title="2.1 Where is demand outstripping supply?",
    problem=("Recruiters need to know which segments have lots of applicants "
             "per opening (competitive) vs few applicants (hard to attract)."),
    solution=("`demand_intensity_score` aggregated by `category_1` × "
              "`title_seniority`. Red = competitive, Blue = hard to attract."),
    chart_fn=_chart_21, insight_fn=_insight_21, action_fn=_action_21,
    sig=SIG,
)


# ─── 2.2 Which roles are hardest to fill? ────────────────────────────────
def _chart_22(sig: tuple) -> None:
    table = _hard_to_fill_table_cached(sig)
    if table.empty:
        st.info("No rows to rank in the current filter set.")
        return
    st.dataframe(
        table,
        width='stretch', hide_index=True,
        column_config={
            "Hard-to-fill": st.column_config.ProgressColumn(
                "Hard-to-fill",
                min_value=float(table["Hard-to-fill"].min()),
                max_value=float(table["Hard-to-fill"].max()),
                format="%.2f",
            ),
            "Apps/vacancy": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption("Mass-hiring rows are excluded by default in the sidebar — toggle off to include them.")


def _insight_22(sig: tuple) -> str:
    table = _hard_to_fill_table_cached(sig).head(20)
    if table.empty:
        return "No hard-to-fill candidates in the current filter."
    top_cat = table["Category"].mode().iloc[0] if not table["Category"].mode().empty else "—"
    mean_duration = table["Duration (days)"].mean()
    mean_reposts = table["Reposts"].mean()
    return (f"Top 20 hard-to-fill roles cluster in **{top_cat}**. "
            f"Mean duration {mean_duration:.0f} days, "
            f"average {mean_reposts:.1f} reposts.")


def _action_22(sig: tuple) -> str:
    return ("For these roles consider direct sourcing, broader years-of-experience "
            "requirements, or a compensation review.")


render_analysis_section(
    title="2.2 Which roles are hardest to fill?",
    problem="Some postings stay open for months and need repeated reposting — surface them.",
    solution=("`hard_to_fill_score` combines posting_duration_days, repostCount, "
              "and inverse applications_per_vacancy (z-scored, clipped ±5)."),
    chart_fn=_chart_22, insight_fn=_insight_22, action_fn=_action_22,
    sig=SIG,
)


# ─── 2.3 Salary positioning ──────────────────────────────────────────────
def _chart_23(sig: tuple) -> None:
    sub = _salary_top_cats(sig)
    if sub.empty:
        st.plotly_chart(empty_chart(), width='stretch')
        return
    fig = px.box(
        sub, x="category_1", y="average_salary",
        color_discrete_sequence=[PALETTE["primary"]],
        labels={"category_1": "", "average_salary": "Average salary (clipped at S$25k)"},
        points=False,
    )
    fig.update_layout(xaxis_tickangle=-30, height=440)
    st.plotly_chart(themed(fig), width='stretch')


def _insight_23(sig: tuple) -> str:
    res = _salary_spread_top(sig)
    if res is None:
        return "Not enough rows for salary positioning."
    cat, p25, p75 = res
    return (f"Within **{cat}**, the 75th-percentile salary is "
            f"S${p75:,.0f}. Roles paying below S${p25:,.0f} sit in the bottom quartile.")


def _action_23(sig: tuple) -> str:
    return "Benchmark your offers against this distribution before posting."


render_analysis_section(
    title="2.3 Salary positioning",
    problem='"Is my S$X offer competitive for a {category} {seniority} role?"',
    solution=("Box plot of `average_salary` by `category_1` (top 12 by count). "
              "Use the calculator below for an explicit segment benchmark."),
    chart_fn=_chart_23, insight_fn=_insight_23, action_fn=_action_23,
    sig=SIG,
)

# Salary calculator widget
with st.container(border=True):
    st.markdown("**Salary benchmark calculator**")
    with st.form("salary_calc"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cat = st.selectbox("Category",
                               sorted(df_f["category_1"].dropna().astype(str).unique()))
        with c2:
            sen_options = [s for s in
                           ["C-suite", "Director/Head", "Manager/Lead", "Senior", "Mid/Other", "Junior"]
                           if s in df_f["title_seniority"].astype(str).unique()]
            sen = st.selectbox("Seniority", sen_options)
        with c3:
            yoe = st.slider("Years of experience", 0, 20, 3)
        submitted = st.form_submit_button("Calculate benchmark")
    if submitted:
        mask = (
            (df_f["category_1"].astype(str) == cat)
            & (df_f["title_seniority"].astype(str) == sen)
            & (df_f["minimumYearsExperience"] == yoe)
        )
        seg = df_f.loc[mask, "average_salary"].dropna()
        if seg.empty:
            st.info("No postings match this exact segment. Try widening the filter or seniority/YoE.")
        else:
            p25 = seg.quantile(0.25)
            p50 = seg.median()
            p75 = seg.quantile(0.75)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sample size", fmt_int(len(seg)))
            m2.metric("P25", fmt_sgd(p25))
            m3.metric("Median", fmt_sgd(p50))
            m4.metric("P75", fmt_sgd(p75))


# ─── 2.4 Engagement funnel ───────────────────────────────────────────────
def _chart_24(sig: tuple) -> None:
    points = _scatter_sample(sig)
    if points.empty:
        st.plotly_chart(empty_chart(), width='stretch')
        return
    full_n = len(get_filtered_df(sig))
    fig = px.scatter(
        points,
        x="metadata_totalNumberOfView", y="metadata_totalNumberJobApplication",
        color="apps_per_view_vs_category_median",
        color_continuous_scale=DIVERGING_SCALE,
        range_color=(-0.05, 0.05),
        labels={"metadata_totalNumberOfView": "Views",
                "metadata_totalNumberJobApplication": "Applications",
                "apps_per_view_vs_category_median": "Conv. vs cat. median"},
        opacity=0.55,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(
        height=480,
        coloraxis_colorbar=dict(
            title=dict(text="Conversion vs<br>category median",
                       font=dict(size=12, color=PALETTE["text"])),
            tickfont=dict(size=11, color=PALETTE["text"]),
            tickformat=".2f",
        ),
    )
    st.plotly_chart(themed(fig), width='stretch')
    if full_n > 5000:
        st.caption(f"Sampled 5,000 of {full_n:,} rows for chart performance.")
    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "Each dot is one posting. **X-axis** = total views, **Y-axis** "
            "= total applications. **Red dots** convert *below* their "
            "category's median (`apps_per_view` lower than peers); **blue "
            "dots** convert *above*. Look for **red dots far to the right** "
            "with low Y — postings that pulled traffic but failed to "
            "convert. Those are JD-quality or fit problems, not visibility "
            "problems."
        )


def _insight_24(sig: tuple) -> str:
    n = _scatter_underperformer_count(sig)
    if n is None:
        return "Not enough engagement data."
    return f"**{n:,}** postings have above-median views but below-median application rates."


def _action_24(sig: tuple) -> str:
    return "Audit JD clarity, title specificity, or stated requirements for these underperformers."


render_analysis_section(
    title="2.4 Engagement funnel",
    problem='"My posting got views but few applications — why?"',
    solution=("Scatter views × applications, coloured by "
              "`apps_per_view_vs_category_median`. Red = converts below "
              "category norm."),
    chart_fn=_chart_24, insight_fn=_insight_24, action_fn=_action_24,
    sig=SIG,
)


# ─── 2.5 Hidden gems ─────────────────────────────────────────────────────
def _chart_25(sig: tuple) -> None:
    table = _hidden_gems_table_cached(sig)
    if table.empty:
        st.info("No hidden gems in the current filter.")
        return
    st.dataframe(
        table, width='stretch', hide_index=True,
        column_config={
            "Conv. rate": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def _insight_25(sig: tuple) -> str:
    res = _hidden_gems_summary(sig)
    if res is None:
        return "No hidden gems found."
    n, top = res
    return f"**{n:,}** hidden-gem postings identified. **{top}** dominates."


def _action_25(sig: tuple) -> str:
    return "Review JD language from these for what works in niche markets."


render_analysis_section(
    title="2.5 Hidden gems",
    problem=("Some postings are quiet but well-targeted — high conversion "
             "despite low traffic. Worth studying."),
    solution="`hidden_gem_flag` = bottom 25% views AND top 25% conversion.",
    chart_fn=_chart_25, insight_fn=_insight_25, action_fn=_action_25,
    sig=SIG,
)


# ─── 2.6 Agency vs direct employer ───────────────────────────────────────
def _chart_26(sig: tuple) -> None:
    sub = _agency_vs_direct_sample(sig).copy()
    sub["employer_type"] = sub["is_agency"].map({True: "Agency", False: "Direct"})
    fig = go.Figure().set_subplots(
        rows=2, cols=2,
        subplot_titles=("Salary distribution", "Conversion rate",
                        "Posting duration", "Seniority mix"),
    )

    for emp, color in [("Direct", PALETTE["primary"]), ("Agency", PALETTE["warning"])]:
        s = sub[sub["employer_type"] == emp]
        if s.empty:
            continue
        fig.add_trace(
            go.Box(x=s["average_salary"].clip(upper=25_000), name=emp,
                   marker_color=color, showlegend=False, boxmean=True),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(x=s["applications_per_view"].clip(upper=0.5), name=emp,
                   marker_color=color, showlegend=False, boxmean=True),
            row=1, col=2,
        )
        fig.add_trace(
            go.Box(x=s["posting_duration_days"], name=emp,
                   marker_color=color, showlegend=False, boxmean=True),
            row=2, col=1,
        )
        sen_order = ["C-suite", "Director/Head", "Manager/Lead", "Senior", "Mid/Other", "Junior"]
        mix = s["title_seniority"].astype(str).value_counts(normalize=True)
        mix = mix.reindex(sen_order).fillna(0)
        fig.add_trace(
            go.Bar(x=mix.index, y=mix.values, name=emp,
                   marker_color=color, showlegend=True),
            row=2, col=2,
        )

    fig.update_layout(height=540, barmode="group")
    st.plotly_chart(themed(fig), width='stretch')


def _insight_26(sig: tuple) -> str:
    s = _agency_vs_direct_stats(sig)
    pct = s["agency_share"]
    med_dir, med_ag = s["median_direct"], s["median_agency"]
    if pd.notna(med_dir) and pd.notna(med_ag):
        diff_text = fmt_sgd(med_ag - med_dir)
    else:
        diff_text = "—"
    return f"Agencies post **{pct:.1%}** of roles. Median salary difference: **{diff_text}**."


def _action_26(sig: tuple) -> str:
    return ("Filter to direct employers when benchmarking salaries for "
            "comparable roles.")


render_analysis_section(
    title="2.6 Agency vs direct employer",
    problem=("Recruitment agencies post differently from end-employers. "
             "Aggregate stats can mislead."),
    solution="`is_agency` flag (detected from company-name keywords).",
    chart_fn=_chart_26, insight_fn=_insight_26, action_fn=_action_26,
    sig=SIG,
)
