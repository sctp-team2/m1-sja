"""Cached data loader, file-upload dispatcher, and shared
filter-application logic.

`load_features()` is the single entry point for the feature frame —
every page calls it. By default it reads `data/mcf_features.pkl`. If
the user uploads a file through the sidebar (`render_data_source_picker`),
that file is parsed, fed through `build_features` if it has the raw
21-column schema, and used in place of the default.

`apply_filters(df, filters)` consumes the dict produced by
`render_sidebar_filters`. Pages should not filter manually.
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# solutions/data/mcf_features.pkl relative to this file
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "mcf_features.pkl"

# Public Drive link to the sample feature pkl. Surfaced as a download
# button in the sidebar so users grab the file locally, then upload it
# back through the file_uploader. This avoids in-app downloads of
# ~220 MB and the flakiness of Drive's virus-scan interstitial.
SAMPLE_DATA_DRIVE_URL = (
    "https://drive.google.com/file/d/1GXoN9DYIZUt3XlPvLVaG2RQktkn1zKO9/view"
)

# Allow `from feature_engineering import build_features` when the app
# is run from inside `app/` (Streamlit sets sys.path to the app dir).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Feature columns the app actually reads. Used to detect a complete
# feature frame; an upload missing any of these is rebuilt or rejected.
_REQUIRED_FEATURE_COLS = (
    "is_agency", "salary_band", "hard_to_fill_score",
    "demand_intensity_score", "title_seniority",
    "zero_engagement_flag", "mass_hiring_flag",
    "salary_suspicious_low", "average_salary", "category_1",
    "metadata_originalPostingDate", "minimumYearsExperience",
)

# Minimum raw columns required to run `build_features` on an upload.
_RAW_REQUIRED = (
    "metadata_originalPostingDate", "metadata_newPostingDate",
    "metadata_expiryDate", "metadata_repostCount",
    "metadata_totalNumberJobApplication", "metadata_totalNumberOfView",
    "salary_minimum", "salary_maximum", "average_salary",
    "numberOfVacancies", "minimumYearsExperience",
    "title", "postedCompany_name", "category_1", "positionLevels",
)


def _parse_upload(content: bytes, name: str) -> pd.DataFrame:
    """Read a user-uploaded file into a DataFrame based on extension."""
    ext = name.rsplit(".", 1)[-1].lower()
    buf = io.BytesIO(content)
    if ext in ("pkl", "pickle"):
        return pd.read_pickle(buf)
    if ext == "csv":
        return pd.read_csv(buf, low_memory=False)
    if ext == "parquet":
        return pd.read_parquet(buf)
    raise ValueError(f"Unsupported file extension: {ext}")


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a fully-featured frame. If `df` already has every column
    the app needs, return as-is. If it has the raw schema, rebuild
    features. Otherwise raise with the specific gap."""
    missing_features = [c for c in _REQUIRED_FEATURE_COLS if c not in df.columns]
    if not missing_features:
        return df
    missing_raw = [c for c in _RAW_REQUIRED if c not in df.columns]
    if not missing_raw:
        from feature_engineering import build_features  # imported lazily
        return build_features(df)
    raise ValueError(
        "Uploaded data is incomplete. Missing "
        f"{len(missing_features)} feature column(s) including: "
        + ", ".join(missing_features[:5])
        + ". Re-export from a current `feature_engineering.build_features` "
        "run, or upload the raw 21-column MCF extract instead."
    )


@st.cache_data(show_spinner="Loading default feature data…")
def _load_default() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature pickle not found at {DATA_PATH}. "
            "Build it first with: "
            "python -c \"import pandas as pd; from feature_engineering import build_features; "
            "build_features(pd.read_pickle('../data/clean_job_step1.pkl'))"
            ".to_pickle('data/mcf_features.pkl')\""
        )
    return pd.read_pickle(DATA_PATH)


@st.cache_data(show_spinner="Parsing uploaded file…", max_entries=4)
def _load_upload(content: bytes, name: str) -> pd.DataFrame:
    """Parse + (optionally) feature-engineer an uploaded file. Cached by
    file content so repeated reads are free."""
    df = _parse_upload(content, name)
    return _ensure_features(df)


def data_source_key() -> str:
    """Short stable token identifying the active data source.

    Returned to callers so they can disambiguate cache keys when the
    user swaps the dataset. `default` for the built-in pkl; otherwise
    `upload:<name>:<sha1[:8]>`.
    """
    upload = st.session_state.get("uploaded_file")
    if upload is None:
        return "default"
    return f"upload:{upload['name']}:{upload['hash']}"


def load_features() -> pd.DataFrame:
    """Return the active feature frame.

    Uses an uploaded file if one is present in `st.session_state`,
    otherwise the bundled pkl at `data/mcf_features.pkl`.
    """
    upload = st.session_state.get("uploaded_file")
    if upload is not None:
        return _load_upload(upload["content"], upload["name"])
    return _load_default()


def load_features_or_stop() -> pd.DataFrame:
    """Like `load_features`, but on Streamlit Cloud (or any host where the
    bundled pkl is absent) show a friendly prompt and halt the page render
    instead of crashing with a traceback."""
    try:
        return load_features()
    except FileNotFoundError:
        st.info(
            "📂 No dataset loaded yet. Use **🌐 Load from URL** "
            "or upload a file in the sidebar to get started."
        )
        st.stop()
    except ValueError as e:
        st.error(f"📂 {e}")
        st.stop()


def _set_session_upload(content: bytes, name: str) -> None:
    """Store an upload in session_state with a hash for cache discrimination."""
    digest = hashlib.sha1(content).hexdigest()[:8]
    existing = st.session_state.get("uploaded_file")
    if not existing or existing["hash"] != digest:
        st.session_state["uploaded_file"] = {
            "content": content,
            "name": name,
            "hash": digest,
        }


def render_data_source_picker() -> None:
    """Sidebar widget for swapping the active data source.

    Place this above `render_sidebar_filters` on every page so changing
    the source refreshes the filters' option lists.

    Two paths to a dataset:
      1. Bundled `data/mcf_features.pkl` (the default; used when nothing
         else is loaded — only present when running locally).
      2. Download the sample pkl from Google Drive (link below), then
         upload it back via `st.file_uploader` (.pkl / .csv / .parquet).
    """
    sb = st.sidebar
    sb.markdown("### Data source")

    sb.markdown(
        f"**Step 1.** [⬇️ Download sample dataset]({SAMPLE_DATA_DRIVE_URL}) "
        "from Google Drive (~220 MB)."
    )
    sb.markdown("**Step 2.** Upload the downloaded file below:")

    uploaded = sb.file_uploader(
        "Upload .pkl / .csv / .parquet",
        type=["pkl", "pickle", "csv", "parquet"],
        help=(
            "Raw 21-column files are auto-passed through `build_features`. "
            "Pre-built 67-column feature files are used as-is."
        ),
        key="data_source_uploader",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        _set_session_upload(uploaded.getvalue(), uploaded.name)

    # ── Status caption + clear button ────────────────────────────────
    active = st.session_state.get("uploaded_file")
    if active is not None:
        try:
            df = load_features()
            sb.success(
                f"Using **{active['name']}** — {len(df):,} rows × {df.shape[1]} cols",
                icon="📂",
            )
            if sb.button("Revert to bundled file", width="stretch"):
                st.session_state.pop("uploaded_file", None)
                st.session_state.pop("data_source_uploader", None)
                st.rerun()
        except Exception as e:
            sb.error(f"Could not load active dataset: {e}")
            st.session_state.pop("uploaded_file", None)
    else:
        sb.caption(
            "Using bundled `data/mcf_features.pkl`. "
            "Click *Load from URL* or upload a file to override."
        )


def filter_signature(filters: dict) -> tuple:
    """Hashable, deterministic encoding of the filter dict.

    Used as a cache key for `get_filtered_df` and downstream cached
    aggregations. The leading element is the data-source key so cache
    entries don't collide when the user swaps the underlying dataset.
    """
    def _conv(v):
        if isinstance(v, list):
            return ("L", tuple(v))
        if isinstance(v, tuple):
            return ("T", v)  # already hashable; preserve element types
        return v
    src = ("__source__", data_source_key())
    return (src,) + tuple(sorted((k, _conv(v)) for k, v in filters.items()))


def _from_signature(sig: tuple) -> dict:
    out = {}
    for k, v in sig:
        if k == "__source__":
            continue  # source key is for cache discrimination only
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
