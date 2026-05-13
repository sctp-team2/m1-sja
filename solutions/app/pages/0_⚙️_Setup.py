"""Page 0 — Setup.

Configure the data source and execution engine. Selections persist
across the rest of the app via @st.cache_data / @st.cache_resource
and st.session_state — visit this page once at the start of a session
or whenever you want to swap inputs.

Kept separate from the analysis pages so filtering and setup don't
share the sidebar.
"""

from __future__ import annotations

import streamlit as st

from lib.data_loader import (
    render_data_source_picker, render_data_status,
    render_execution_mode_toggle,
)

st.set_page_config(page_title="Setup · HR Recruiter Insights", layout="wide")

st.title("⚙️ Setup")
st.caption(
    "Pick the dataset and the execution engine. Both selections "
    "persist across pages — no need to revisit unless you want to "
    "change something."
)

col_src, col_mode = st.columns(2)
with col_src:
    render_data_source_picker()
with col_mode:
    render_execution_mode_toggle()

st.divider()
render_data_status()
