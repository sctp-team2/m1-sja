"""HR Recruiter Insights — landing page.

Entry point for the Streamlit multi-page app. Streamlit auto-discovers
the files in `pages/` and renders the sidebar navigation. This module
itself is the landing page: it shows the headline narrative, a few
quick-orient KPIs, and explains how to use the rest of the app.

Run from the `solutions/` directory:
    streamlit run app/app.py
"""

from __future__ import annotations

import streamlit as st

from lib.chart_helpers import PALETTE, fmt_int, fmt_sgd
from lib.data_loader import (
    engine_badge, filter_signature, get_filtered_df, load_features_or_stop,
    render_data_source_picker, render_execution_mode_toggle,
)
from lib.filters import filter_summary, render_sidebar_filters

st.set_page_config(
    page_title="HR Recruiter Insights",
    page_icon="📊",
    layout="wide",
)

st.title("HR Recruiter Insights")
st.caption(
    f"Singapore job-market analytics for HR analysts and recruiters. "
    f"Built on MyCareersFuture postings. · Engine: **{engine_badge()}**"
)

render_execution_mode_toggle()
render_data_source_picker()
df = load_features_or_stop()
filters = render_sidebar_filters(df)
df_f = get_filtered_df(filter_signature(filters))

st.sidebar.markdown("---")
st.sidebar.markdown(filter_summary(filters, df_f, df))

st.markdown("### Use the sidebar to navigate")
st.markdown(
    """
- **📊 Overview** — five-second orientation: KPIs, top categories,
  top companies, salary and seniority mix.
- **🔍 Deep Analysis** — where demand outstrips supply, which roles
  are hardest to fill, salary positioning, engagement funnels, agency
  vs direct.
- **📋 Recruitment Report** — the operational deliverable. Promising
  roles, category recommendations, salary benchmarks, action list,
  and CSV exports.
- **🧰 Tools** — interactive recruiter utilities. Salary benchmark
  calculator today; more on request.
"""
)

st.markdown("### Right now")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Postings in view", fmt_int(len(df_f)),
          delta=f"{len(df_f) / len(df):.0%} of total" if len(df) else None,
          delta_color="off")
c2.metric("Median salary", fmt_sgd(df_f["average_salary"].median()) if len(df_f) else "—")
c3.metric(
    "Top category",
    str(df_f["category_1"].mode().iloc[0]) if len(df_f) else "—",
)
c4.metric(
    "Active employers",
    fmt_int(df_f["postedCompany_name"].nunique()) if len(df_f) else "—",
)

if len(df_f) == 0:
    st.warning("No postings match the current filters. Reset filters in the sidebar to continue.")

st.markdown("---")
st.markdown(
    f"<span style='color:{PALETTE['muted']}; font-size: 0.9rem;'>"
    "Data: pre-built feature frame (67 columns) derived from MCF posting data. "
    "See <code>feature_engineering.py</code> for the pipeline."
    "</span>",
    unsafe_allow_html=True,
)
