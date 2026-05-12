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
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

# solutions/data/mcf_features.pkl relative to this file
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "mcf_features.pkl"

# Sample dataset hosted on Google Drive. Pre-fills the URL input so a
# fresh clone can fetch the data with one click — the user can edit
# the URL if they have a different remote source.
SAMPLE_DATA_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=18k_90SR01cQXFFRPLgPGZ3Y3YnDe0r3x&export=download&confirm=t"
)

# Allow `from feature_engineering import build_features` when the app
# is run from inside `app/` (Streamlit sets sys.path to the app dir).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Columns that prove an input is already feature-engineered. If any of
# these appear in the upload, skip `build_features`.
_FEATURE_MARKERS = ("hard_to_fill_score", "salary_band", "demand_intensity_score")

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
    """If `df` is already feature-engineered, return as-is. Else run
    `build_features` on it. Raises if the schema fits neither shape."""
    if any(col in df.columns for col in _FEATURE_MARKERS):
        return df
    missing = [c for c in _RAW_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded file is neither a feature frame nor a recognised raw "
            "MCF schema. Missing required raw columns: "
            + ", ".join(missing[:6])
            + (f" (+{len(missing) - 6} more)" if len(missing) > 6 else "")
        )
    from feature_engineering import build_features  # imported lazily
    return build_features(df)


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


def _extract_drive_id(url: str) -> str | None:
    """Return the Google Drive file ID embedded in a URL, if any.

    Handles the common shareable patterns:
      - https://drive.google.com/file/d/<ID>/view
      - https://drive.google.com/uc?id=<ID>
      - https://drive.usercontent.google.com/download?id=<ID>&...
    """
    if "drive.google.com" not in url and "drive.usercontent.google.com" not in url:
        return None
    m = re.search(r"/file/d/([A-Za-z0-9_-]{20,})", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", url)
    return m.group(1) if m else None


@st.cache_data(show_spinner="Downloading from Google Drive…", max_entries=2)
def _download_from_drive(file_id: str) -> tuple[bytes, str]:
    """Fetch a Google-Drive-hosted file by its ID using `gdown`.

    Handles the virus-scan interstitial Google returns for files >25 MB.
    Returns (content, original_filename). Cached on `file_id`.
    """
    import gdown  # lazy import — large dep, only needed for this code path
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = str(Path(tmpdir)) + "/"
        path = gdown.download(id=file_id, output=out_dir, quiet=True)
        if not path:
            raise RuntimeError(
                "Google Drive returned no file. Check the file is shared "
                "with 'Anyone with the link' and the ID is correct."
            )
        path = Path(path)
        with open(path, "rb") as f:
            content = f.read()
        return content, path.name


@st.cache_data(show_spinner="Downloading from URL…", max_entries=2)
def _download_from_url(url: str) -> tuple[bytes, str]:
    """Generic HTTP fetch with filename detection.

    Filename priority:
      1. `Content-Disposition` header.
      2. Path component of the URL (e.g. `.../mcf_features.pkl`).
      3. Fallback `download.pkl`.

    Returns (content, filename). Cached on `url`.
    """
    import requests
    with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", "").lower():
            raise RuntimeError(
                "URL returned HTML, not a binary file. For Google Drive "
                "shareable URLs, paste the file ID or use a direct-download "
                "URL containing `confirm=t`."
            )
        content = r.content
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
        if m:
            name = m.group(1)
        else:
            parsed_name = Path(urlparse(url).path).name
            name = parsed_name if "." in parsed_name else "download.pkl"
    return content, name


def _fetch_url(url: str) -> tuple[bytes, str]:
    """Dispatch to the Drive-specific path when the URL looks like Drive,
    otherwise fall back to a generic HTTP GET."""
    drive_id = _extract_drive_id(url)
    if drive_id:
        return _download_from_drive(drive_id)
    return _download_from_url(url)


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

    Three paths to a dataset:
      1. Bundled `data/mcf_features.pkl` (the default; used when nothing
         else is loaded).
      2. URL load — editable text input pre-filled with the sample
         Google Drive URL; the "Load from URL" button fetches it.
      3. Local file upload via `st.file_uploader`
         (.pkl / .csv / .parquet).
    """
    sb = st.sidebar
    sb.markdown("### Data source")

    # ── Option B: load from URL (editable, pre-filled with sample) ──
    st.session_state.setdefault("data_source_url", SAMPLE_DATA_URL)
    with sb.expander("🌐 Load from URL", expanded=True):
        st.text_input(
            "Source URL",
            key="data_source_url",
            help=(
                "Google Drive shareable URLs are auto-detected by file ID. "
                "Other public URLs are fetched directly. Edit to point at "
                "your own dataset."
            ),
        )
        if st.button("Load from URL", width="stretch", type="primary"):
            try:
                content, name = _fetch_url(st.session_state["data_source_url"])
                _set_session_upload(content, name)
                # Clear the local-file uploader so its 'None' on the next
                # rerun doesn't fight with the URL-installed upload.
                st.session_state.pop("data_source_uploader", None)
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")

    # ── Option C: upload from local disk ─────────────────────────────
    uploaded = sb.file_uploader(
        "…or upload locally (.pkl / .csv / .parquet)",
        type=["pkl", "pickle", "csv", "parquet"],
        help=(
            "Raw 21-column files are auto-passed through `build_features`. "
            "Pre-built 67-column feature files are used as-is."
        ),
        key="data_source_uploader",
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
