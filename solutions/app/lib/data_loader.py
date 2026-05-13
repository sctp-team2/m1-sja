"""Cached data loader, file-upload dispatcher, and shared
filter-application logic.

`load_features()` is the single entry point for the feature frame —
every page calls it. By default it reads `data/m1-eda-clean-v1.pkl`. If
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

import duckdb
import pandas as pd
import streamlit as st

# Pandas-mode disk-size threshold for the "consider DuckDB" warning.
_PANDAS_MODE_WARN_BYTES = 500 * 1024 * 1024  # 500 MB per upgrade-v2.md spec

# solutions/data/m1-eda-clean-v1.pkl relative to this file
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "m1-eda-clean-v1.pkl"

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
            ".to_pickle('data/m1-eda-clean-v1.pkl')\""
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
    otherwise the bundled pkl at `data/m1-eda-clean-v1.pkl`.
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
    """Store an upload in session_state with a hash for cache discrimination.

    When the hash changes (genuine new upload, not a re-render of the
    same file), drop every @st.cache_data entry so old per-page
    aggregations and the previous source frame are released. The
    DuckDB connection (cache_resource) survives — its `features` view
    is re-registered to the new frame on the next _query_duckdb call.
    """
    digest = hashlib.sha1(content).hexdigest()[:8]
    existing = st.session_state.get("uploaded_file")
    if not existing or existing["hash"] != digest:
        st.session_state["uploaded_file"] = {
            "content": content,
            "name": name,
            "hash": digest,
        }
        st.cache_data.clear()


def _active_source_size_bytes() -> int | None:
    """Disk size of the active source, in bytes. None when unknown."""
    upload = st.session_state.get("uploaded_file")
    if upload is not None:
        return len(upload["content"])
    if DATA_PATH.exists():
        return DATA_PATH.stat().st_size
    return None


def engine_badge() -> str:
    """Short label for the active execution engine; for page captions."""
    return "DuckDB" if st.session_state.get("use_duckdb", False) else "pandas"


def render_data_status() -> None:
    """Sidebar diagnostics: where is the data right now?

    Surfaces the three orthogonal pieces of state a user might wonder
    about: which source file is active, whether the pandas frame is
    loaded into the @st.cache_data store, and whether the DuckDB
    connection is live with a registered view. Architecturally the pkl
    is always read into pandas first; DuckDB sees a zero-copy view of
    that same frame (current bridge build).
    """
    sb = st.sidebar
    with sb.expander("Data status", expanded=False):
        # Source
        upload = st.session_state.get("uploaded_file")
        size = _active_source_size_bytes()
        size_str = f"{size / (1024**2):.0f} MB" if size else "unknown size"
        if upload is not None:
            st.markdown(f"**Source:** upload `{upload['name']}` · {size_str}")
        else:
            st.markdown(f"**Source:** bundled `{DATA_PATH.name}` · {size_str}")

        # Pandas frame state — every page calls load_features() before
        # this widget renders, so by now the frame is hot in cache.
        try:
            df = load_features()
            st.markdown(f"**In-memory frame:** loaded · {len(df):,} × {df.shape[1]}")
        except Exception:
            st.markdown("**In-memory frame:** not loaded yet")

        # DuckDB connection / view state. Only meaningful in DuckDB mode.
        if st.session_state.get("use_duckdb", False):
            con = get_duckdb_con()
            view_row = con.execute(
                "SELECT count(*) FROM duckdb_views() WHERE view_name = 'features'"
            ).fetchone()
            view_bound = bool(view_row and view_row[0])
            if view_bound:
                row_count = con.execute("SELECT count(*) FROM features").fetchone()[0]
                st.markdown(
                    f"**DuckDB:** connection live · view `features` bound · "
                    f"{row_count:,} rows visible"
                )
            else:
                st.markdown(
                    "**DuckDB:** connection live · view `features` not yet "
                    "registered (no query has run in this mode)"
                )
        else:
            st.markdown("**DuckDB:** idle (toggle off — pandas filters in use)")

        st.caption(
            "Architecture: pkl → pandas (cached) → either pandas mask "
            "or DuckDB SQL on a registered view of the same frame. "
            "DuckDB never reads pkl directly."
        )


def render_execution_mode_toggle() -> None:
    """Sidebar widget pair: DuckDB/pandas toggle + clear-cache button.

    Place at the very top of the sidebar on every page so it sits above
    the data-source picker and filters. Mode persists in
    `st.session_state["use_duckdb"]` and is folded into the cache key
    by `filter_signature`, so each mode keeps its own cached results.

    A 500 MB pandas-mode warning fires when the active source exceeds
    the threshold and the toggle is OFF — nudges the user toward
    DuckDB without forcing a switch.
    """
    sb = st.sidebar
    sb.markdown("### Execution mode")
    sb.checkbox(
        "Load into DuckDB",
        key="use_duckdb",
        help=(
            "OFF: pandas filters the in-memory frame (simple, predictable). "
            "ON: DuckDB runs SQL against a registered view of the same "
            "frame — same columns out, same display code."
        ),
    )
    if st.session_state.get("use_duckdb", False):
        sb.caption(
            "**DuckDB mode** · SQL engine on the loaded frame. "
            "Pickle source is bridged through pandas — switch to .parquet "
            "for true predicate pushdown."
        )
    else:
        sb.caption(
            "**Pandas mode** · simpler, holds the full dataset in memory. "
            "Toggle on for DuckDB SQL semantics."
        )

    # 500 MB warning — only relevant when pandas mode is loading a large file.
    size = _active_source_size_bytes()
    if (
        not st.session_state.get("use_duckdb", False)
        and size is not None
        and size > _PANDAS_MODE_WARN_BYTES
    ):
        sb.warning(
            f"Active source is {size / (1024**2):.0f} MB — over the "
            f"{_PANDAS_MODE_WARN_BYTES // (1024**2)} MB pandas-mode "
            "threshold. Consider toggling **Load into DuckDB**.",
            icon="⚠️",
        )

    if sb.button("Clear cache", width="stretch", help="Clears @st.cache_data; mode toggle is preserved."):
        st.cache_data.clear()
        st.toast("Cache cleared.", icon="🧹")


def render_data_source_picker() -> None:
    """Sidebar widget for swapping the active data source.

    Place this above `render_sidebar_filters` on every page so changing
    the source refreshes the filters' option lists.

    Two paths to a dataset:
      1. Bundled `data/m1-eda-clean-v1.pkl` (the default; used when nothing
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
                st.cache_data.clear()
                st.rerun()
        except Exception as e:
            sb.error(f"Could not load active dataset: {e}")
            st.session_state.pop("uploaded_file", None)
    else:
        sb.caption(
            "Using bundled `data/m1-eda-clean-v1.pkl`. "
            "Click *Load from URL* or upload a file to override."
        )


def filter_signature(filters: dict) -> tuple:
    """Hashable, deterministic encoding of the filter dict.

    Used as a cache key for `get_filtered_df` and downstream cached
    aggregations. The leading elements are the data-source key and the
    execution mode so cache entries don't collide when the user swaps
    the underlying dataset or toggles DuckDB on/off.
    """
    def _conv(v):
        if isinstance(v, list):
            return ("L", tuple(v))
        if isinstance(v, tuple):
            return ("T", v)  # already hashable; preserve element types
        return v
    src = ("__source__", data_source_key())
    mode = ("__mode__", "duckdb" if st.session_state.get("use_duckdb", False) else "pandas")
    return (src, mode) + tuple(sorted((k, _conv(v)) for k, v in filters.items()))


def _from_signature(sig: tuple) -> dict:
    out = {}
    for k, v in sig:
        if k in ("__source__", "__mode__"):
            continue  # discrimination keys only; not real filters
        if isinstance(v, tuple) and len(v) == 2 and v[0] in ("L", "T"):
            out[k] = list(v[1]) if v[0] == "L" else v[1]
        else:
            out[k] = v
    return out


def _mode_from_signature(sig: tuple) -> str:
    for k, v in sig:
        if k == "__mode__":
            return v
    return "pandas"


@st.cache_resource(show_spinner=False)
def get_duckdb_con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection, shared across reruns and sessions.

    Created lazily on first DuckDB-mode query. The active feature frame
    is registered as a view named `features` per-query (see
    `_query_duckdb`) — pickle is not natively readable by DuckDB, so we
    bridge through pandas: pandas owns the in-memory frame, DuckDB runs
    SQL against it via zero-copy view registration.
    """
    return duckdb.connect(database=":memory:", read_only=False)


def _build_duckdb_where(filters: dict) -> tuple[str, list]:
    """Compose a parameterised WHERE clause from the filter dict.

    Returns (where_sql, params). `where_sql` is empty when no filters
    apply; otherwise it starts with " WHERE ". Placeholder count for
    `IN (...)` is generated from the value count — values themselves
    are bound, never f-stringed.
    """
    clauses: list[str] = []
    params: list = []

    date_range = filters.get("date_range")
    if date_range and len(date_range) == 2:
        clauses.append("metadata_originalPostingDate BETWEEN ? AND ?")
        params.extend([pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])])

    def _in_clause(col: str, values: list) -> None:
        if not values:
            return
        placeholders = ",".join(["?"] * len(values))
        clauses.append(f"CAST({col} AS VARCHAR) IN ({placeholders})")
        params.extend(values)

    _in_clause("category_1", filters.get("categories") or [])
    _in_clause("title_seniority", filters.get("seniorities") or [])
    _in_clause("salary_band", filters.get("salary_bands") or [])
    _in_clause("employmentTypes", filters.get("employment_types") or [])

    yoe_range = filters.get("yoe_range")
    if yoe_range and yoe_range != (0, 20):
        clauses.append("minimumYearsExperience BETWEEN ? AND ?")
        params.extend([yoe_range[0], yoe_range[1]])

    if filters.get("exclude_zero_engagement"):
        clauses.append("NOT zero_engagement_flag")
    if filters.get("exclude_mass_hiring"):
        clauses.append("NOT mass_hiring_flag")
    if filters.get("exclude_suspicious_low"):
        clauses.append("NOT salary_suspicious_low")

    employer = filters.get("employer_type", "Both")
    if employer == "Direct only":
        clauses.append("NOT is_agency")
    elif employer == "Agency only":
        clauses.append("is_agency")

    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


def _restore_pandas_dtypes(out: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """Coerce DuckDB output dtypes back to the pandas-path dtypes so
    downstream display code is mode-agnostic.

    DuckDB's `.df()` returns category cols as object and dates as
    datetime64 — but loses the `category` dtype the pages occasionally
    rely on. Restore by reusing the source frame's dtype map.
    """
    for col in out.columns:
        if col not in src.columns:
            continue
        src_dtype = src[col].dtype
        if str(src_dtype) == "category" and str(out[col].dtype) != "category":
            out[col] = pd.Categorical(
                out[col],
                categories=src[col].cat.categories,
                ordered=src[col].cat.ordered,
            )
        elif str(src_dtype).startswith("datetime") and not str(out[col].dtype).startswith("datetime"):
            out[col] = pd.to_datetime(out[col])
    return out


def _query_pandas(sig: tuple) -> pd.DataFrame:
    """Pandas-mode query path: boolean-mask filter on the in-memory frame."""
    df = load_features()
    return apply_filters(df, _from_signature(sig))


def _query_duckdb(_con: duckdb.DuckDBPyConnection, sig: tuple) -> pd.DataFrame:
    """DuckDB-mode query path: register the active frame as a view, run
    parameterised SQL, return a DataFrame.

    Note on this build: DuckDB runs the filter SQL, but the source
    pickle is held by pandas (DuckDB cannot read .pkl natively). The
    benefit here is SQL semantics + DuckDB's vectorised filter engine,
    not avoiding the pandas load. Switch the source file to .parquet to
    unlock predicate pushdown without the bridge.
    """
    df = load_features()
    # register() is REPLACE semantics — re-binding "features" to the
    # current frame each call ensures DuckDB sees the active upload,
    # never a stale one from a previous file.
    _con.register("features", df)
    where_sql, params = _build_duckdb_where(_from_signature(sig))
    sql = f"SELECT * FROM features{where_sql}"
    out = _con.execute(sql, params).df()
    out = _restore_pandas_dtypes(out, df)
    assert set(out.columns) == set(df.columns), (
        "DuckDB path returned a different column set than the source frame "
        "— schema parity broken"
    )
    return out


@st.cache_data(show_spinner="Filtering…", max_entries=8)
def get_filtered_df(sig: tuple) -> pd.DataFrame:
    """Return the filtered feature frame, cached by filter signature.

    Dispatches to the pandas or DuckDB path based on the mode embedded
    in `sig` (set by `filter_signature` from `st.session_state`). Both
    paths return DataFrames with the same columns and dtypes so the
    display layer never branches on mode.
    """
    if _mode_from_signature(sig) == "duckdb":
        return _query_duckdb(get_duckdb_con(), sig)
    return _query_pandas(sig)


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
