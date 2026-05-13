"""Page 1 — Overview.

Five-second orientation to the Singapore job market for an HR analyst
arriving with no specific question in mind. Answers: how big is the
market, who's posting most, where's the pay, and what's the seniority
mix.

Consumes: category_1, postedCompany_name, average_salary,
positionLevels, employmentTypes, title_seniority, salary_band,
is_agency, is_reposted, metadata_originalPostingDate,
posting_duration_days.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.chart_helpers import (
    AGENCY_COLORS, PALETTE, annotation_style, empty_chart, fmt_int,
    fmt_sgd, themed,
)
from lib.data_loader import (
    filter_signature, get_filtered_df, load_features_or_stop,
    render_data_source_picker,
)
from lib.filters import filter_summary, render_sidebar_filters

st.set_page_config(page_title="Overview · HR Recruiter Insights", layout="wide")

render_data_source_picker()
df = load_features_or_stop()
filters = render_sidebar_filters(df)
df_f = get_filtered_df(filter_signature(filters))
st.sidebar.markdown("---")
st.sidebar.markdown(filter_summary(filters, df_f, df))

st.title("Singapore Job Market Overview")
if len(df_f):
    d_min = df_f["metadata_originalPostingDate"].min().date()
    d_max = df_f["metadata_originalPostingDate"].max().date()
    st.caption(f"Showing **{len(df_f):,}** postings from **{d_min}** to **{d_max}**.")
else:
    st.warning("No postings match current filters. Reset filters in the sidebar.")
    st.stop()


# ─── KPI strip ────────────────────────────────────────────────────────────
median_salary = df_f["average_salary"].median()
top_category = df_f["category_1"].mode().iloc[0] if not df_f["category_1"].mode().empty else "—"
median_duration = df_f["posting_duration_days"].median()
repost_share = df_f["is_reposted"].mean()

# Baseline (unfiltered) for delta context
all_median_salary = df["average_salary"].median()
all_median_duration = df["posting_duration_days"].median()
all_repost_share = df["is_reposted"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total postings", fmt_int(len(df_f)),
          delta=f"of {len(df):,} total", delta_color="off")
k2.metric("Median salary", fmt_sgd(median_salary),
          delta=fmt_sgd(median_salary - all_median_salary) if pd.notna(median_salary) else None)
k3.metric("Top category", str(top_category))
k4.metric("Median posting duration", f"{int(median_duration)} days" if pd.notna(median_duration) else "—",
          delta=f"{int(median_duration - all_median_duration):+d} vs all" if pd.notna(median_duration) else None,
          delta_color="off")
k5.metric("Reposted share", f"{repost_share:.1%}",
          delta=f"{(repost_share - all_repost_share):+.1%} vs all",
          delta_color="off")

st.markdown("---")


# ─── Top categories | Top companies ───────────────────────────────────────
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("Top 15 categories")
    st.caption("Posting count by `category_1`.")
    cat_counts = (
        df_f["category_1"].value_counts().head(15).sort_values(ascending=True)
    )
    if cat_counts.empty:
        st.plotly_chart(empty_chart(), width="stretch")
    else:
        fig = px.bar(
            x=cat_counts.values, y=cat_counts.index.astype(str),
            orientation="h",
            labels={"x": "Postings", "y": ""},
            color_discrete_sequence=[PALETTE["primary"]],
        )
        fig.update_traces(hovertemplate="%{y}<br>%{x:,} postings<extra></extra>")
        fig.update_layout(height=480, margin=dict(l=180))
        st.plotly_chart(themed(fig), width="stretch")
        with st.expander("ℹ️ How to read this chart"):
            st.markdown(
                "**Bars** show the top 15 MCF job categories by raw posting "
                "count in the current filter. **Use this to** spot which "
                "industries dominate the market — large bars mean recruiters "
                "and end-employers post heavily here, so competition for "
                "talent is high. Combine with the salary distribution below "
                "to see whether high posting volume comes with high pay."
            )

with c_right:
    st.subheader("Top 15 companies")
    st.caption("Posting count by `postedCompany_name`. Color: agency vs direct.")
    co = (
        df_f.groupby("postedCompany_name", observed=True)
        .agg(postings=("metadata_jobPostId", "count"),
             is_agency=("is_agency", "max"))
        .reset_index()
        .sort_values("postings", ascending=False)
        .head(15)
        .sort_values("postings", ascending=True)
    )
    if co.empty:
        st.plotly_chart(empty_chart(), width="stretch")
    else:
        co["employer_type"] = co["is_agency"].map(
            {True: "Agency", False: "Direct"}
        )
        fig = px.bar(
            co, x="postings", y="postedCompany_name", color="employer_type",
            orientation="h",
            color_discrete_map={"Agency": PALETTE["warning"], "Direct": PALETTE["primary"]},
            labels={"postings": "Postings", "postedCompany_name": "",
                    "employer_type": "Employer type"},
        )
        fig.update_traces(hovertemplate="%{y}<br>%{x:,} postings<extra></extra>")
        fig.update_layout(height=480, margin=dict(l=200))
        st.plotly_chart(themed(fig), width="stretch")
        with st.expander("ℹ️ How to read this chart"):
            st.markdown(
                "**Bars** show the 15 companies with the most postings. "
                "**Orange = recruitment agency**, **blue = direct employer** "
                "(`is_agency` is a keyword match on company name). "
                "Agencies often re-list the same role under different titles, "
                "so a high agency count doesn't mean high net hiring. For "
                "clean salary benchmarks, use the sidebar to filter to "
                "*Direct only*."
            )

st.markdown("---")


# ─── Salary section ───────────────────────────────────────────────────────
st.subheader("Salary distribution")
sal = df_f["average_salary"].clip(upper=25_000)
sal = sal[sal.notna()]

if sal.empty:
    st.plotly_chart(empty_chart(), width="stretch")
else:
    p50 = sal.median()
    p75 = sal.quantile(0.75)
    fig = px.histogram(
        sal, nbins=50, color_discrete_sequence=[PALETTE["primary"]],
        labels={"value": "Average salary (SGD/month, clipped at S$25k)"},
    )
    # Distinct positions so the two pills don't overlap when p50 and p75
    # are close. "top left" anchors the pill to the left of its line.
    fig.add_vline(
        x=p50, line_dash="dash", line_color=PALETTE["critical"], line_width=2.5,
        annotation_text=f"<b>Median</b><br>S${int(p50):,}",
        annotation_position="top left",
        annotation=annotation_style(PALETTE["critical"]),
    )
    fig.add_vline(
        x=p75, line_dash="dot", line_color=PALETTE["warning"], line_width=2.5,
        annotation_text=f"<b>P75</b><br>S${int(p75):,}",
        annotation_position="top right",
        annotation=annotation_style(PALETTE["warning"]),
    )
    fig.update_traces(hovertemplate="S$%{x}<br>%{y:,} postings<extra></extra>")
    fig.update_layout(
        showlegend=False, yaxis_title="Postings", height=460,
        margin=dict(t=110, l=60, r=60, b=50),
    )
    st.plotly_chart(themed(fig), width="stretch")
    with st.expander("ℹ️ How to read this chart"):
        # `\$` escape — Streamlit markdown otherwise treats `$X$` as LaTeX.
        st.markdown(
            "Each bar counts how many postings fall in a S\\$500 salary "
            "bucket. The **red dashed line** is the median; the **amber "
            "dotted line** is the 75th percentile. Bars to the right of "
            "P75 sit in the top quartile. **Note:** salaries above "
            "S\\$25k are clipped into the rightmost bar — MCF caps "
            "disclosed salary at ~S\\$20k/month, so very senior roles "
            "are right-censored and the upper tail is unreliable."
        )

st.subheader("Salary by position level")
st.caption("Box plot of `average_salary` by `positionLevels`. Sample size annotated.")
if len(df_f) > 0 and df_f["positionLevels"].notna().any():
    pos_med = (
        df_f.dropna(subset=["positionLevels"])
        .groupby("positionLevels", observed=True)["average_salary"]
        .median()
        .sort_values()
    )
    counts = df_f["positionLevels"].value_counts()
    fig = go.Figure()
    for lvl in pos_med.index:
        sub = df_f.loc[df_f["positionLevels"] == lvl, "average_salary"].clip(upper=30_000)
        # Two-line name: tier + sample size. Plotly renders `<br>` cleanly
        # in trace names; an inline `<span>` doesn't, so keep it simple.
        fig.add_trace(go.Box(
            x=sub, name=f"{lvl}<br>n={counts.get(lvl, 0):,}",
            marker_color=PALETTE["primary"], boxmean=True, orientation="h",
        ))
    fig.update_layout(
        xaxis_title="Average salary (SGD/month, clipped at S$30k)",
        yaxis_title="", showlegend=False, height=520,
        margin=dict(l=240, r=20, t=20, b=50),
    )
    fig.update_yaxes(tickfont=dict(size=12))
    st.plotly_chart(themed(fig), width="stretch")
    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "Each row is a `positionLevels` tier, ordered by median pay. "
            "**The box** spans P25 to P75; the **line inside** is the "
            "median; the **diamond** marks the mean. **Whiskers** extend "
            "to ~1.5× IQR; points beyond are outliers. The `n=` next to "
            "each tier name is how many postings sit in that tier — small "
            "samples (e.g. Top Management) have wider, less reliable boxes."
        )

st.markdown("---")


# ─── Mix donuts ───────────────────────────────────────────────────────────
d_left, d_right = st.columns(2)

with d_left:
    st.subheader("Employment type mix")
    et = df_f["employmentTypes"].astype(str).value_counts().head(10)
    if et.empty:
        st.plotly_chart(empty_chart(), width="stretch")
    else:
        fig = px.pie(values=et.values, names=et.index, hole=0.5,
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_traces(
            textposition="outside",
            textinfo="label+percent",
            textfont=dict(size=13, color=PALETTE["text"]),
            insidetextorientation="radial",
            hovertemplate="%{label}<br>%{value:,} (%{percent})<extra></extra>",
            marker=dict(line=dict(color="white", width=2)),
        )
        fig.update_layout(showlegend=False, height=380,
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(themed(fig), width="stretch")
        with st.expander("ℹ️ How to read this chart"):
            st.markdown(
                "Slice size = share of postings with each "
                "`employmentTypes` value. Permanent and Full-Time dominate "
                "MCF; freelance and part-time are minor segments. Filter "
                "to a specific type in the sidebar to see how each "
                "segment's salary distribution differs."
            )

with d_right:
    st.subheader("Seniority mix")
    sen_order = ["C-suite", "Director/Head", "Manager/Lead", "Senior", "Mid/Other", "Junior"]
    sen = df_f["title_seniority"].astype(str).value_counts()
    sen = sen.reindex([s for s in sen_order if s in sen.index])
    if sen.empty:
        st.plotly_chart(empty_chart(), width="stretch")
    else:
        fig = px.pie(values=sen.values, names=sen.index, hole=0.5,
                     color_discrete_sequence=px.colors.sequential.Viridis_r)
        fig.update_traces(
            textposition="outside",
            textinfo="label+percent",
            textfont=dict(size=13, color=PALETTE["text"]),
            insidetextorientation="radial",
            hovertemplate="%{label}<br>%{value:,} (%{percent})<extra></extra>",
            marker=dict(line=dict(color="white", width=2)),
        )
        fig.update_layout(showlegend=False, height=380,
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(themed(fig), width="stretch")
        with st.expander("ℹ️ How to read this chart"):
            st.markdown(
                "Slice size = share of postings whose **title** signals "
                "each seniority (`title_seniority`, ordered C-suite → "
                "Junior). The 'Mid/Other' slice is a catch-all for titles "
                "without an explicit signal — its share doesn't mean "
                "all those roles are mid-level. Compare against the "
                "position-level box plot above for a second view."
            )
