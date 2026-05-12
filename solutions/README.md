# MCF Recruiter Insights Dashboard

A Streamlit dashboard for HR analysts and recruiters analysing
MyCareersFuture (MCF) Singapore job-posting data. The dashboard answers
one composite business question:

> *"For the segment I care about, which roles are most worth
> prioritizing — high vacancy count, manageable competition, fair salary
> expectations — and what should I do differently to fill them?"*

Built on a pre-computed 67-column feature frame derived from 1,012,928
MCF postings (Mar 2023 – snapshot date).

## What's in this directory

```
solutions/
├── feature_engineering.py            10-block pipeline, raw → 46 derived columns
├── m1-eda-step2-analyze.ipynb        existing EDA notebook (step 2)
├── m1-feature-engineering-step3.ipynb walkthrough of the pipeline + validation
├── recruiter-dashboard.md            full app build spec
├── skills.md                         developer orientation
├── CLAUDE.md                         AI-assistant operating manual
├── REPORT.md                         written report (Sections 1–4)
├── README.md                         this file
├── requirements.txt
├── data/
│   └── mcf_features.pkl              built feature frame (~236 MB, gitignored)
└── app/
    ├── app.py                        landing page
    ├── lib/                          shared modules (loader, filters, charts, suggestions)
    └── pages/                        Streamlit multi-page entries
        ├── 1_📊_Overview.py
        ├── 2_🔍_Deep_Analysis.py
        └── 3_📋_Recruitment_Report.py
```

The raw data file `../data/clean_job_step1.pkl` (1M+ rows, 21 columns)
is the input to the pipeline. It is **not committed** because of size —
see [Data](#data) below for how to obtain it.

## Quick start

### 1. Environment

A conda env named `pds` already configured for the wider course is the
expected setup. If you have it:

```bash
conda activate pds
pip install streamlit plotly                # one-time
```

From scratch:

```bash
conda create -n pds python=3.10 -y
conda activate pds
pip install -r requirements.txt
```

Python 3.10+ is required — the pipeline uses `X | None` type hints.

### 2. Data

The raw MCF posting data lives at `../data/clean_job_step1.pkl`
(relative to this directory). It is the cleaned output of a
prior step-1 notebook. Download / regenerate as appropriate, then:

```bash
# Build the feature frame the app consumes (~5 seconds on 1M rows)
python -c "
import pandas as pd
from feature_engineering import build_features
build_features(pd.read_pickle('../data/clean_job_step1.pkl')) \
    .to_pickle('data/mcf_features.pkl')
"
```

This produces `data/mcf_features.pkl` (~236 MB, 1,012,928 × 67).

### 3. Run the dashboard

```bash
streamlit run app/app.py
```

The app opens at <http://localhost:8501>. First load reads and caches
the pickle (~3 seconds); subsequent navigation is instant.

### 4. (Optional) Run the notebook

```bash
jupyter lab m1-feature-engineering-step3.ipynb
```

Walks through each pipeline block on a small sample, shows derived
columns, and reproduces the validation checks.

## The three pages at a glance

| Page | Question it answers |
| --- | --- |
| **📊 Overview** | What does the market look like — categories, companies, salary spread, seniority mix? |
| **🔍 Deep Analysis** | Where does demand outstrip supply? Which roles are hardest to fill? Is my offer competitive? |
| **📋 Recruitment Report** | Operational summary: top promising roles, category recommendations, salary matrix, action list, CSV exports. |

Every chart has an `ℹ️ How to read this chart` expander, collapsed by
default — click to reveal a plain-English explanation of what the
visual is showing and how to use it.

## Shared filters

The sidebar persists across all pages. Filters write to
`st.session_state`, so navigating doesn't lose context:

- Date range, category, seniority, salary band, years of experience,
  employment type
- Data-quality toggles: exclude zero-engagement postings (on by
  default), exclude mass-hiring postings (on by default), exclude
  suspiciously-low salaries (off by default)
- Employer type radio: Both / Direct only / Agency only

"Reset filters" restores every default.

## How the data is built

1. **Raw** — `../data/clean_job_step1.pkl` (21 columns, 1M+ rows)
   produced upstream from `SGJobData.csv` by an EDA pipeline.
2. **`feature_engineering.build_features(df_raw)`** runs 10 ordered
   blocks (time → salary → engagement → title → company → quality flags
   → within-category normalization → recency → company enrichments →
   composite scores). Every block is a pure function — no in-place
   mutation — so the pipeline is re-runnable in isolation.
3. **Output** — `data/mcf_features.pkl` (67 columns: 21 raw + 46
   derived). This is the single source the dashboard reads.

See `m1-feature-engineering-step3.ipynb` for a step-by-step view.

## Architecture notes

- **`feature_engineering.py`** — the data layer. Do not modify casually;
  if you do, re-run the validation block in `skills.md`.
- **`app/lib/data_loader.py`** — `load_features()` is cached via
  `@st.cache_data`, so the 236 MB pickle reads once per session and is
  shared across pages.
- **`app/lib/filters.py`** — single sidebar widget. State persists in
  `st.session_state`; navigating pages preserves selections.
- **`app/lib/suggestions.py`** — per-posting recommendation engine.
  Powers the "Recruiter action list" on Page 3 Section 3.5.
- **`app/lib/chart_helpers.py`** — Plotly palette, theming, annotation
  styles. All charts go through `themed(fig)`.

## Known data quirks

These are baked into the pipeline assumptions — important if you
extend the dashboard:

1. **MCF salary ceiling at ~S$20k/month.** Senior/exec roles are
   right-censored; upper-tail benchmarks are unreliable.
2. **`salary_undisclosed_flag` is 0% on this dataset.** Defensive only.
3. **Mass-hiring rows (~1.5%)** with `numberOfVacancies > 10` distort
   per-posting stats — excluded by default in the sidebar.
4. **Agency postings are ~21%.** Use "Direct only" for clean salary
   benchmarks.
5. **Composite scores are z-scored and clipped to ±5.**
6. **DataFrame index has gaps.** Do not assume contiguous integers.

## Out of scope (v1)

Listed for clarity, not built. See `recruiter-dashboard.md` §12.

- Skill-tag extraction from titles (Python, AWS, SAP, …)
- Geographic features (Singapore districts from titles)
- PDF report export
- Saved filter presets, comparison mode, auth, live scraping

## Acceptance criteria

Mirrors `recruiter-dashboard.md` §11:

1. All three pages load without errors on the full dataset.
2. Sidebar filters propagate across pages and persist via
   `st.session_state`.
3. Page 2 sections follow Problem → Solution → Chart → Insight →
   Action; Insight and Action update with filters.
4. Page 3 §3.2 "Promising Roles" matches the spec definition.
5. Suggestion engine produces feature-traceable recommendations.
6. CSV exports work on Page 3.
7. Color palette and typography are consistent.
8. Page load <3 s after the cache warms.
9. App handles empty filter results.

## Troubleshooting

**"Feature pickle not found at data/mcf_features.pkl"** — the build
step in [Data](#data) wasn't run. Re-run it.

**"st.session_state has no key 'date_range'"** — stale session state
from an old build. Clear it: stop Streamlit, refresh the browser,
restart with `streamlit run app/app.py`.

**Slow first page load** — the 236 MB pickle is being read. Subsequent
loads are cached for the session.

**Plotly chart deprecation warnings** — none should fire with
Streamlit 1.40+ and Plotly 5.20+. If they do, update both packages.
