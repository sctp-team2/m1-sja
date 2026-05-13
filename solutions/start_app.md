# Starting the MCF Recruiter Insights App

Two deploy paths, both end at the same dashboard:

- **A. Local** — fastest iteration; uses your bundled / built feature
  pickle directly.
- **B. Streamlit Community Cloud** — public URL, no local install;
  data is pulled from a Google Drive URL the first time the app boots.

If anything below doesn't work, check `README.md` first for repo
layout and architecture context.

---

## A. Run locally

### 1. Get the code

```bash
git clone <your-fork-url>
cd <repo>/6m-data-1.5.1-onboarding/my-work/m1-sja/solutions
```

All commands below assume you're in this `solutions/` directory.

### 2. Set up Python 3.10+

Either reuse the course conda env:

```bash
conda activate pds
pip install -r requirements.txt
```

Or create a fresh venv:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Required versions and the reason each is pinned:

| Package | Min | Why |
| --- | --- | --- |
| `streamlit` | 1.40 | Native multi-page apps, `st.column_config.ProgressColumn`, `st.fragment` |
| `plotly` | 5.20 | Cleaner default theming, `update_coloraxes` |
| `pandas` | 2.0 | `observed=True` default change, `DataFrame.map` (replaces `applymap`) |
| `numpy` | 1.24 | (transitive — pandas) |
| `pyarrow` | 14 | Parquet read/write (optional once everything moved to `.pkl`) |
| `gdown` | 5.0 | Google Drive download with virus-scan token handling |
| `requests` | 2.28 | Non-Drive URL fetches in the data-source picker |

### 3. Provide a feature frame — pick one

**Option (i) — auto-fetch on first run.** Skip ahead. Launch the
app, then click **🌐 Load from URL** in the sidebar. The pre-filled
URL points at a Google Drive copy of `mcf_features.pkl` (~236 MB,
downloads in 5–10 s on a decent connection). The app caches the
result for the rest of the session.

**Option (ii) — build the feature pickle locally** if you already
have the cleaned step-1 frame at `../data/clean_job_step1.pkl`:

```bash
python -c "
import pandas as pd
from feature_engineering import build_features
build_features(pd.read_pickle('../data/clean_job_step1.pkl')) \
    .to_pickle('data/mcf_features.pkl')
"
```

This writes `solutions/data/mcf_features.pkl` (~236 MB). The app
prefers this local file over the Drive URL when both are available.

**Option (iii) — your own data.** Use the sidebar's **upload local
file** widget on any page. Accepts `.pkl` / `.csv` / `.parquet`.
Raw 21-column MCF schema → auto-runs `build_features`. Pre-built
67-column feature frame → used as-is.

### 4. Launch

```bash
streamlit run app/app.py
```

Streamlit opens the app at <http://localhost:8501> (port may
auto-increment if 8501 is busy). First page load reads + caches the
feature frame (1–3 s). Subsequent navigation is near-instant.

### 5. (Optional) Run the data-handling notebook

```bash
jupyter lab m1-feature-engineering-step3.ipynb
```

Walks through every pipeline block on a stratified sample with the
validation checks (shape == 67, NaN audit, within-category z-scores).

### Useful local commands

| Goal | Command |
| --- | --- |
| Run on a custom port | `streamlit run app/app.py --server.port 8765` |
| Run headless (no auto-browser) | `streamlit run app/app.py --server.headless true` |
| Disable telemetry | `streamlit run app/app.py --browser.gatherUsageStats false` |
| Smoke-test the pipeline only | `python feature_engineering.py` |
| Validate every page renders | `python -c "from streamlit.testing.v1 import AppTest; [print(AppTest.from_file(p).run().exception or 'ok') for p in ['app/app.py','app/pages/1_📊_Overview.py','app/pages/2_🔍_Deep_Analysis.py','app/pages/3_📋_Recruitment_Report.py','app/pages/4_🧰_Tools.py']]"` |

---

## B. Deploy on Streamlit Community Cloud

[share.streamlit.io](https://share.streamlit.io) is the free hosting
option. The feature pickle is too big to commit to git (~236 MB), so
the cloud build expects the user to click **Load from URL** in-app to
fetch from Google Drive on first boot.

### 1. Push the repo to GitHub

The deploy target is the `solutions/` directory; everything else in
the larger course repo is optional context. Two common layouts:

**a. Whole course repo (already what you have).** Push as-is. In
Streamlit Cloud's "Main file path" field you'll specify
`6m-data-1.5.1-onboarding/my-work/m1-sja/solutions/app/app.py`.

**b. Solutions-only repo (cleaner).** Create a smaller repo that
mirrors only `solutions/`. Main file path becomes `app/app.py`.

Files that **must** be in the deployed tree:

```
<repo-root>/                                  (or .../solutions/ if extracted)
├── feature_engineering.py
├── requirements.txt
└── app/
    ├── app.py
    ├── lib/                  data_loader.py, filters.py, chart_helpers.py,
    │                         suggestions.py, __init__.py
    └── pages/                1_📊_Overview.py, 2_🔍_Deep_Analysis.py,
                              3_📋_Recruitment_Report.py, 4_🧰_Tools.py
```

`data/mcf_features.pkl` must **not** be committed (it's gitignored and
huge — pulled at runtime via the Drive URL instead).

### 2. (Optional) Pin the Python version

Streamlit Cloud defaults to Python 3.13. The pipeline works on 3.10+
but you can pin explicitly by creating a `runtime.txt` at the repo
root (or alongside `requirements.txt` in solutions-only layout):

```
python-3.11
```

3.10 also works; 3.13 has been tested but install times for some
binary wheels are longer.

### 3. Configure the app

On <https://share.streamlit.io/new>:

| Field | Value |
| --- | --- |
| Repository | `<your-github-username>/<repo-name>` |
| Branch | `main` |
| Main file path | `solutions/app/app.py` (or `app/app.py` for option b) |
| App URL | autogenerated, editable in Advanced settings |
| Python version | 3.10+ (3.11 recommended) |

No secrets are needed — the Drive URL is public and the app reads no
private API keys. If you later swap to a private dataset, add the URL
to `secrets.toml` and read it via `st.secrets["DATA_URL"]`.

### 4. First boot on the deployed app

1. Wait for the build to finish (~3–5 min the first time —
   `gdown` and `pyarrow` are the slow installs).
2. Open the app URL. The sidebar shows the data-source picker.
3. Click **Load from URL**. The default URL is the sample feature
   pickle on Google Drive; the download takes 5–10 s on Streamlit
   Cloud, then the file is cached for the session.
4. Navigate pages 📊 / 🔍 / 📋 / 🧰 freely — the cached frame is
   shared across all of them.

If you want to bypass the manual click, you can change
`SAMPLE_DATA_URL` in `app/lib/data_loader.py` to your own URL and
modify `load_features()` to call `_fetch_url(SAMPLE_DATA_URL)` when
no upload is in `st.session_state`. (Not done by default to keep
the deploy explicit.)

### 5. Memory & performance on the free tier

Streamlit Community Cloud's free tier has ~1 GB RAM. The
236 MB feature pickle loads to ~700–900 MB peak pandas memory
including categoricals and intermediate aggregates. This is tight:

- **Filter changes** are fine — refilter + cached aggs total 1–2 s.
- **Page 3 "Prepare filtered CSV"** can OOM on the full dataset
  (~750 MB temp string buffer). The button is opt-in for this
  reason — don't click it unless you've narrowed the filter set.
- **Going viral?** Upgrade to a Teams/Pro plan or self-host. The app
  is stateless apart from `st.session_state`, so horizontal scaling
  is trivial.

### 6. Troubleshooting deployment

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Build error "No module named …" | Missing dep in `requirements.txt` | Add it, push, redeploy |
| `gdown.download(…) returned None` | Drive file no longer shared "Anyone with link" | Re-share the file, or paste a new URL in the sidebar |
| `KeyError: 'st.session_state has no key "date_range"'` | Stale session after a code update | Click "Reboot app" in Streamlit Cloud manage panel |
| App boots but says "Feature pickle not found" | User hasn't clicked Load yet | Expected — click 🌐 Load from URL |
| OOM crash on Page 3 | Full-filter CSV pre-encoded | Filter narrower; the lazy CSV button only encodes on click |
| Slow first paint | Cold cache + Drive download | Normal on first session; <2 s after warmup |

---

## C. Bonus — Docker (advanced)

A minimal `Dockerfile` for self-hosting:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY feature_engineering.py .
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--browser.gatherUsageStats", "false"]
```

Build and run:

```bash
docker build -t mcf-insights .
docker run --rm -p 8501:8501 mcf-insights
```

Inside the container, click **Load from URL** to fetch the sample
data — or mount your own `data/mcf_features.pkl` with `-v
$(pwd)/data:/app/data`.

---

## Related docs

- `README.md` — what each file does, architecture notes, data quirks.
- `REPORT.md` — written report (Sections 1–4) for the assignment.
- `recruiter-dashboard.md` — full app build spec.
- `m1-feature-engineering-step3.ipynb` — data-handling walkthrough.
