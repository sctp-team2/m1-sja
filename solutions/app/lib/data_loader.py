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

try:
    import psutil  # optional; observability degrades gracefully if absent
except ImportError:
    psutil = None  # type: ignore

# Pandas-mode disk-size threshold for the "consider DuckDB" warning.
_PANDAS_MODE_WARN_BYTES = 500 * 1024 * 1024  # 500 MB per upgrade-v2.md spec

# Default-data candidates, tried in order. First existing wins.
# Parquet is preferred when present: smaller on disk, column pruning
# when DuckDB mode reads it directly (future), and faster pandas load.
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_DEFAULT_CANDIDATES: tuple[Path, ...] = (
    _DATA_DIR / "eda_step3_data.parquet",
    _DATA_DIR / "m1-eda-clean-v1.pkl",
)


def _resolve_default_path() -> Path | None:
    """Return the first candidate that exists on disk, else None."""
    for p in _DEFAULT_CANDIDATES:
        if p.exists():
            return p
    return None


def _default_path_for_display() -> Path:
    """Resolved default if present, otherwise the preferred candidate
    (useful for status strings even when neither file exists yet)."""
    return _resolve_default_path() or _DEFAULT_CANDIDATES[0]


# Back-compat alias for code that still references DATA_PATH. Resolves
# at call time so a file appearing later in the session is picked up.
class _DataPathProxy:
    def __getattr__(self, item):
        return getattr(_default_path_for_display(), item)

    def __fspath__(self) -> str:
        return str(_default_path_for_display())

    def __str__(self) -> str:
        return str(_default_path_for_display())


DATA_PATH = _DataPathProxy()


def _read_data_file(path: Path) -> pd.DataFrame:
    """Format-dispatched read for a file path (mirrors _parse_upload)."""
    ext = path.suffix.lower().lstrip(".")
    if ext in ("pkl", "pickle"):
        return pd.read_pickle(path)
    if ext == "csv":
        return pd.read_csv(path, low_memory=False)
    if ext == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported default-file extension: {ext} ({path})")

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
    path = _resolve_default_path()
    if path is None:
        candidates = " or ".join(p.name for p in _DEFAULT_CANDIDATES)
        raise FileNotFoundError(
            f"No default data file present. Looked for {candidates} "
            f"under {_DATA_DIR}/. Drop one in, or upload via the "
            "Setup page."
        )
    df = _read_data_file(path)
    # Raw uploads get auto-engineered; do the same for raw defaults so
    # a freshly-dropped parquet of step-1 data still becomes a feature
    # frame the rest of the app expects.
    return _ensure_features(df)


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
            "📂 No dataset loaded yet. Open the **⚙️ Setup** page "
            "(left sidebar nav) to upload one."
        )
        if st.button("Go to ⚙️ Setup"):
            st.switch_page("pages/0_⚙️_Setup.py")
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
        # Drop any materialized DuckDB table — it now reflects stale data.
        try:
            drop_duckdb_state(get_duckdb_con())
        except Exception:
            pass


def _active_source_size_bytes() -> int | None:
    """Disk size of the active source, in bytes. None when unknown."""
    upload = st.session_state.get("uploaded_file")
    if upload is not None:
        return len(upload["content"])
    p = _resolve_default_path()
    if p is not None:
        return p.stat().st_size
    return None


def engine_badge() -> str:
    """Short label for the active execution engine; for page captions."""
    return "DuckDB" if st.session_state.get("use_duckdb", False) else "pandas"


def _human_bytes(n: float | int | None) -> str:
    """Compact byte formatter: 226 MB, 1.2 GB, etc."""
    if n is None:
        return "—"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@st.cache_data(show_spinner=False)
def _frame_deep_memory(sig_for_source: str) -> int:
    """deep=True memory of the active frame; cached by data source key
    so swapping uploads invalidates without re-walking the same frame."""
    df = load_features()
    return int(df.memory_usage(deep=True).sum())


def _duckdb_total_memory(con: duckdb.DuckDBPyConnection) -> int | None:
    """Sum of bytes across all DuckDB subsystems for this connection.

    Returns None when the duckdb_memory() function isn't available or
    fails (older versions, edge runtimes)."""
    try:
        row = con.execute(
            "SELECT COALESCE(sum(memory_usage_bytes), 0) FROM duckdb_memory()"
        ).fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def render_data_status() -> None:
    """Setup-page observability block.

    Surfaces source, memory (DataFrame / Streamlit process / system),
    pipeline state (in-memory frame, dtype mix, date range), DuckDB
    connection state, and environment versions. Architecturally the
    pkl is always read into pandas first; DuckDB sees a zero-copy view
    of that same frame (current bridge build).
    """
    with st.expander("Data status", expanded=True):
        # ── Refresh button — recomputes live metrics on next render ──
        c_title, c_btn = st.columns([5, 1])
        c_title.markdown("Live process / system / DuckDB metrics below.")
        if c_btn.button(
            "🔄 Refresh",
            help="Re-fetch process RSS, system memory, and DuckDB engine memory.",
        ):
            # _frame_deep_memory is keyed on source key, so it stays
            # cached unless the source changed. The rerun re-reads
            # psutil + duckdb_memory() which aren't cached.
            st.rerun()

        # ── Source ───────────────────────────────────────────────────
        upload = st.session_state.get("uploaded_file")
        src_bytes = _active_source_size_bytes()
        src_size_str = _human_bytes(src_bytes)
        if upload is not None:
            st.markdown(f"**Source:** upload `{upload['name']}` · {src_size_str}")
        else:
            st.markdown(f"**Source:** bundled `{_default_path_for_display().name}` · {src_size_str}")

        # ── Memory metrics (the four the operator most often needs) ──
        st.markdown("**Memory**")
        try:
            df_bytes = _frame_deep_memory(data_source_key())
        except Exception:
            df_bytes = None

        proc_rss = None
        sys_total = None
        sys_avail = None
        sys_pct = None
        if psutil is not None:
            try:
                proc_rss = psutil.Process().memory_info().rss
                vm = psutil.virtual_memory()
                sys_total, sys_avail, sys_pct = vm.total, vm.available, vm.percent
            except Exception:
                pass

        duck_mem = None
        if st.session_state.get("use_duckdb", False):
            duck_mem = _duckdb_total_memory(get_duckdb_con())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Uploaded file",
            _human_bytes(src_bytes),
            delta=("upload" if upload is not None else "bundled"),
            delta_color="off",
        )
        m2.metric("DataFrame (in-memory)", _human_bytes(df_bytes))
        m3.metric(
            "DuckDB engine",
            _human_bytes(duck_mem) if duck_mem is not None else "—",
            delta=(None if st.session_state.get("use_duckdb", False) else "mode off"),
            delta_color="off",
        )
        m4.metric(
            "Total system RAM",
            _human_bytes(sys_total),
            delta=(
                f"{_human_bytes(sys_avail)} free · {sys_pct:.0f}% used"
                if sys_pct is not None else None
            ),
            delta_color="off",
        )
        st.caption(
            f"This Streamlit process RSS: **{_human_bytes(proc_rss)}**"
            + ("" if psutil is not None else " · install `psutil` for process/system memory")
        )

        # ── Pipeline ─────────────────────────────────────────────────
        st.markdown("**Pipeline**")
        try:
            df = load_features()
            dtype_counts = df.dtypes.astype(str).value_counts()
            dtype_str = " · ".join(f"{n} {dt}" for dt, n in dtype_counts.items())
            date_col = "metadata_originalPostingDate"
            date_range_str = ""
            if date_col in df.columns and len(df):
                d_lo = df[date_col].min()
                d_hi = df[date_col].max()
                date_range_str = (
                    f" · dates {pd.Timestamp(d_lo).date()} → "
                    f"{pd.Timestamp(d_hi).date()}"
                )
            st.markdown(
                f"In-memory frame: **loaded** · {len(df):,} × {df.shape[1]}"
                f"{date_range_str}"
            )
            st.caption(f"Dtypes: {dtype_str}")
        except Exception:
            st.markdown("In-memory frame: **not loaded yet**")

        # ── DuckDB ───────────────────────────────────────────────────
        st.markdown("**DuckDB**")
        if st.session_state.get("use_duckdb", False):
            con = get_duckdb_con()
            materialized = _has_materialized_table(con)
            view_row = con.execute(
                "SELECT count(*) FROM duckdb_views() WHERE view_name = 'features'"
            ).fetchone()
            view_bound = bool(view_row and view_row[0])
            if materialized:
                row_count = con.execute("SELECT count(*) FROM features").fetchone()[0]
                st.markdown(
                    f"**Connection: ACTIVE** · `features` is a **materialized "
                    f"table** · {row_count:,} rows · engine owns the bytes "
                    "(see Engine memory metric)."
                )
            elif view_bound:
                row_count = con.execute("SELECT count(*) FROM features").fetchone()[0]
                st.markdown(
                    f"**Connection: ACTIVE** · `features` is a **view** "
                    f"(zero-copy bridge to pandas) · {row_count:,} rows · "
                    "engine memory will read 0 B until you materialize."
                )
            else:
                st.markdown(
                    "**Connection: ACTIVE** · `features` not yet registered "
                    "(no query has run in this mode). Use the Setup page's "
                    "**📥 Load DataFrame into DuckDB** button to materialize."
                )
        else:
            st.markdown("**Connection: idle** (toggle off — pandas filters in use).")

        # ── Environment ──────────────────────────────────────────────
        import numpy as _np
        st.markdown("**Environment**")
        st.caption(
            f"streamlit {st.__version__} · duckdb {duckdb.__version__} · "
            f"pandas {pd.__version__} · numpy {_np.__version__}"
            + (f" · psutil {psutil.__version__}" if psutil is not None else "")
        )

        st.caption(
            "Architecture: pkl → pandas (cached) → either pandas mask "
            "or DuckDB SQL on a registered view of the same frame. "
            "DuckDB never reads pkl directly."
        )


def render_execution_mode_toggle() -> None:
    """Setup-page widget: DuckDB/pandas toggle + clear-cache button.

    Mode persists in `st.session_state["use_duckdb"]` and is folded
    into the cache key by `filter_signature`, so each mode keeps its
    own cached results. A 500 MB pandas-mode warning fires when the
    active source exceeds the threshold and the toggle is OFF.
    """
    st.markdown("### Execution mode")
    st.checkbox(
        "Load into DuckDB",
        key="use_duckdb",
        help=(
            "OFF: pandas filters the in-memory frame (simple, predictable). "
            "ON: DuckDB runs SQL against a registered view of the same "
            "frame — same columns out, same display code."
        ),
    )
    if st.session_state.get("use_duckdb", False):
        # Force eager connection creation so status reads "connected"
        # even before any query has run.
        con = get_duckdb_con()
        materialized = _has_materialized_table(con)
        if materialized:
            st.caption(
                "**DuckDB mode** · connection live · table `features` "
                "materialized (engine owns its own buffers — visible in "
                "the engine-memory metric)."
            )
        else:
            st.caption(
                "**DuckDB mode** · connection live · view-only bridge "
                "(zero-copy reference to pandas; engine memory reads 0 B "
                "until you materialize)."
            )

        # Materialize button — only useful when not already materialized
        # and a frame is loadable.
        if not materialized:
            if st.button(
                "📥 Load DataFrame into DuckDB",
                help=(
                    "Run CREATE TABLE features AS SELECT * FROM <pandas>. "
                    "DuckDB takes a real copy so engine memory becomes "
                    "non-zero. Pandas df is kept (mode-switching still "
                    "works); click Clear cache to release it."
                ),
            ):
                try:
                    df_src = load_features()
                    n = materialize_duckdb(con, df_src)
                    st.toast(
                        f"Materialized {n:,} rows into DuckDB.",
                        icon="🦆",
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Materialization failed: {e}")
        else:
            if st.button(
                "♻️ Drop DuckDB table",
                help="Release DuckDB's copy; queries fall back to the pandas-bridge view.",
            ):
                drop_duckdb_state(con)
                st.toast("Dropped features table.", icon="🦆")
                st.rerun()
    else:
        st.caption(
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
        st.warning(
            f"Active source is {size / (1024**2):.0f} MB — over the "
            f"{_PANDAS_MODE_WARN_BYTES // (1024**2)} MB pandas-mode "
            "threshold. Consider toggling **Load into DuckDB**.",
            icon="⚠️",
        )

    if st.button("Clear cache", help="Clears @st.cache_data; mode toggle is preserved."):
        st.cache_data.clear()
        st.toast("Cache cleared.", icon="🧹")


def render_data_source_picker() -> None:
    """Setup-page widget for swapping the active data source.

    Two paths to a dataset:
      1. Bundled `data/m1-eda-clean-v1.pkl` (the default; used when
         nothing else is loaded — only present when running locally).
      2. Download the sample pkl from Google Drive (link below), then
         upload it back via `st.file_uploader` (.pkl / .csv / .parquet).
    """
    st.markdown("### Data source")

    st.markdown(
        f"**Step 1.** [⬇️ Download sample dataset]({SAMPLE_DATA_DRIVE_URL}) "
        "from Google Drive (~220 MB)."
    )
    st.markdown("**Step 2.** Upload the downloaded file below:")

    uploaded = st.file_uploader(
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

    # ── Status caption + revert button ───────────────────────────────
    active = st.session_state.get("uploaded_file")
    if active is not None:
        try:
            df = load_features()
            st.success(
                f"Using **{active['name']}** — {len(df):,} rows × {df.shape[1]} cols",
                icon="📂",
            )
            if st.button("Revert to bundled file"):
                st.session_state.pop("uploaded_file", None)
                st.session_state.pop("data_source_uploader", None)
                st.cache_data.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Could not load active dataset: {e}")
            st.session_state.pop("uploaded_file", None)
    else:
        resolved = _resolve_default_path()
        if resolved is not None:
            size_mb = resolved.stat().st_size / (1024 ** 2)
            st.caption(
                f"Using bundled `data/{resolved.name}` · {size_mb:.0f} MB. "
                "Upload a file above to override."
            )
            if st.button(
                "🔄 Load default file",
                help=(
                    "Re-read the bundled file from disk and warm the cache. "
                    "Use this after an app restart or OOM crash to get back "
                    "to a known-good state in one click."
                ),
                type="primary",
            ):
                # 1. Drop every cached frame and DuckDB object so we
                #    don't read stale data after a re-init.
                st.cache_data.clear()
                try:
                    drop_duckdb_state(get_duckdb_con())
                except Exception:
                    pass
                # 2. Eagerly call the loader — populates _load_default's
                #    cache so the next page render is instant.
                with st.spinner(f"Loading `{resolved.name}`…"):
                    try:
                        df = load_features()
                        st.toast(
                            f"Loaded {resolved.name} · {len(df):,} × {df.shape[1]}",
                            icon="📂",
                        )
                    except Exception as e:
                        st.error(f"Load failed: {e}")
                        st.stop()
                st.rerun()
        else:
            cands = " or ".join(f"`data/{p.name}`" for p in _DEFAULT_CANDIDATES)
            st.caption(
                f"No bundled file found ({cands}). Upload one above."
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


def _has_materialized_table(con: duckdb.DuckDBPyConnection) -> bool:
    """True when a real DuckDB table named 'features' exists on this
    connection (not just a registered pandas view)."""
    try:
        row = con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'features'"
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def _drop_features_object(con: duckdb.DuckDBPyConnection) -> None:
    """Drop the 'features' table or view, whichever exists.

    DuckDB errors if you DROP TABLE on a view or DROP VIEW on a table,
    so we discriminate up-front."""
    try:
        is_table = bool(con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'features'"
        ).fetchone()[0])
        is_view = bool(con.execute(
            "SELECT count(*) FROM duckdb_views() WHERE view_name = 'features'"
        ).fetchone()[0])
    except Exception:
        is_table = is_view = False
    if is_table:
        try:
            con.execute("DROP TABLE features")
        except Exception:
            pass
    if is_view:
        try:
            con.execute("DROP VIEW features")
        except Exception:
            pass


def materialize_duckdb(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Copy the active pandas frame into a real DuckDB table.

    Forces DuckDB to allocate its own column buffers (the bridge view
    is zero-copy, so the engine-memory metric reads 0 B until this
    runs). Idempotent — drops any prior table or view first.
    Returns the row count.
    """
    _drop_features_object(con)
    # Temporary alias used only for the COPY, then released.
    con.register("_features_src", df)
    try:
        con.execute("CREATE TABLE features AS SELECT * FROM _features_src")
    finally:
        try:
            con.unregister("_features_src")
        except Exception:
            pass
    row = con.execute("SELECT count(*) FROM features").fetchone()
    return int(row[0]) if row else 0


def drop_duckdb_state(con: duckdb.DuckDBPyConnection) -> None:
    """Drop any materialized table or registered view named 'features'.

    Called when the data source changes — the prior materialized copy
    no longer reflects the active frame."""
    _drop_features_object(con)


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
    if not _has_materialized_table(_con):
        # No real table yet — register a zero-copy view of the active
        # pandas df. register() has REPLACE semantics, so a new upload
        # automatically swaps the binding on the next query.
        _con.register("features", df)
    # Else: a materialized table is in place. Trust it and SELECT from
    # it directly; the materialize path drops the view when copying.
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
