# MCF Jobs Insights — Developer Init

Tight orientation for anyone (human or AI) picking up this project. Assumes Python/pandas fluency. Read top to bottom — under 5 minutes.

---

## What this is

A Singapore MyCareersFuture (MCF) job-posting analytics project. Two deliverables exist in this repo:

1. **`feature_engineering.py`** — production-quality feature pipeline. Takes raw MCF data, returns a 67-column enriched frame (21 raw + 46 derived).
2. **`recruiter_dashboard_spec.md`** — build spec for a 3-page Streamlit app (Overview, Deep Analysis, Recruitment Report) targeting HR analysts and recruiters.

The dashboard itself is **not yet built**. The data pipeline is complete and validated.

---

## Business context (one line)

Help recruiters spot **high-vacancy, low-application** roles and benchmark salaries against within-category peers, using a fully derived feature set.

---

## Data

**Raw schema** (1,012,928 rows, 21 columns): MCF postings with employment type, posting/expiry dates, repost count, application count, view count, vacancies, position level, company, salary min/max/average, status, title, category fields, hybrid flag.

**Pre-built features** (`build_features(df_raw)`) produces:
- Salary: `salary_band`, `salary_zscore_within_category`, `salary_percentile_within_category`, `salary_premium_vs_category_median`, `salary_per_yoe`
- Engagement: `applications_per_view`, `applications_per_vacancy`, `views_per_vacancy`, `views_zscore_within_category`, `apps_per_view_vs_category_median`
- Composite scores (z-scored, clipped ±5): `hard_to_fill_score`, `demand_intensity_score`, `posting_quality_score`
- Flags: `is_agency`, `is_reposted`, `mass_hiring_flag`, `zero_engagement_flag`, `salary_undisclosed_flag`, `salary_suspicious_low`, `seniority_mismatch_flag`, `hidden_gem_flag`
- Time: `posting_year`, `posting_month`, `posting_quarter`, `posting_dow`, `posting_year_month`, `posting_cohort_quarter`, `days_since_posting`, `is_active_at_snapshot`
- Title: `title_seniority` (ordered: C-suite → Junior), `title_word_count`, `title_char_length`
- Company: `company_posting_count`, `company_posting_bucket`, `company_avg_salary`, `company_salary_spread`, `company_category_diversity`, `company_tenure_days`

Full inventory in the docstrings of `feature_engineering.py`.

---

## Quick start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy pyarrow streamlit plotly

# Smoke-test the feature pipeline
python feature_engineering.py        # runs synthetic test, prints 46 derived columns

# Build features on real data (one-off, ~15-30s on 1M rows)
python -c "
import pandas as pd
from feature_engineering import build_features
df_raw = pd.read_parquet('path/to/raw_mcf.parquet')
df = build_features(df_raw)
df.to_parquet('data/mcf_features.parquet')
"

# When the Streamlit app exists:
streamlit run app/app.py
```

---

## File map

```
sctp-dsai/
├── feature_engineering.py        # data pipeline — single source of derived columns
├── recruiter_dashboard_spec.md   # full app build spec; hand to Claude or dev
├── init.md                       # this file
├── data/
│   └── mcf_features.parquet      # built feature frame (gitignored, regenerate as needed)
├── app/                          # Streamlit app — to be built per spec
│   ├── app.py
│   ├── pages/
│   └── lib/
└── (5m-, 6m-data-* folders)      # SCTP coursework, unrelated to the MCF project
```

The coursework folders (`5m-data-*`, `6m-data-*`) are unrelated to the MCF dashboard work. Ignore them unless explicitly working on a lesson.

---

## Architecture

### Feature pipeline (10 blocks, dependency-ordered)

```
add_time_features        ─┐
add_salary_features      │
add_engagement_features  │
add_title_features       │  raw → derived
add_company_features     │
add_quality_flags        ─┘
        │
        ▼
clean_for_benchmarks(df)        # filter outlier/dirty rows → bench_df
        │
        ▼
add_normalized_features(df, bench_df)   # peer stats from bench
add_recency_features(df)
add_company_enrichments(df, bench_df)
add_composite_scores(df, bench_df)      # z-scored, clipped ±5
```

Each block is a **pure function** — input df not mutated, returns a new frame. Re-runnable in isolation.

### Why two-frame design

The full df keeps all rows (recruiters want to inspect noisy/dead/outlier postings). The `bench_df` is the cleaned subset used only for computing peer benchmarks — so a `numberOfVacancies=999` mass-hiring posting doesn't distort the median for "real" roles.

---

## Conventions

**Function signatures.** Every feature block: `add_x_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame`. No in-place mutation.

**Dtypes.** Memory-efficient choices: `int8`/`int16` for counts/ages, `float32` for derived numerics, `bool` for flags, `category` for low-cardinality strings, `datetime64[ns]` for dates.

**Categoricals + groupby.** Always use `observed=True` when grouping on a categorical column (`postedCompany_name`, `category_1`, etc.) to avoid the cartesian-product explosion.

**Null handling.** Composite scores fill NaN with 0 *only* in score components. Tables preserve NaN. Charts treat NaN as "no data" with a footnote, not silently drop.

**Naming.** snake_case for columns. Suffixes: `_flag` (bool), `_score` (z-scored numeric), `_band` / `_bucket` (categorical), `_within_category` (peer-normalized), `_vs_category_median` (raw diff).

---

## Data quirks to know

These were discovered during validation. Bake them into any analysis:

1. **MCF salary ceiling at ~S$20k/month.** Top observed values cluster at S$19,500–19,999.50. Senior management / executive roles are effectively right-censored. Upper-tail benchmarks are unreliable for these segments.

2. **`salary_undisclosed_flag` is 0% on this dataset.** MCF (or the upstream extract) doesn't surface undisclosed salaries here. The flag is defensive — keeps working if the data source changes.

3. **Mass-hiring rows distort raw stats.** Rows with `numberOfVacancies > 10` (1.5% of dataset) have `applications_per_vacancy` values near 0.001 and pull composite-score scales. The clean-benchmark pattern handles this; **always exclude mass hiring from leaderboards by default**.

4. **Agency rows are 21% of dataset.** `is_agency` keyword-based detection: RECRUIT, STAFFING, MANPOWER, etc. Filter for clean salary benchmarks of "real" employers.

5. **Index has gaps.** Raw df has 1,012,928 rows but max index ~1,048,574. Don't assume contiguous integer index.

6. **`seniority_mismatch_flag` fires on 12%.** Position-level vs title-seniority disagreement. Mostly genuine misalignment in MCF data, but the regex is imperfect — treat as a hint, not ground truth.

---

## Common dev tasks

### Add a new feature

1. Decide which block it belongs to (or add a new one).
2. Write as a pure function: `add_X_features(df, **kwargs) -> pd.DataFrame`.
3. If it depends on another block, place it after that block in `build_features` and document the dependency in the docstring.
4. Add it to the smoke test in `if __name__ == "__main__"`.
5. Run `python feature_engineering.py` — it should print the new column with the expected dtype.

### Re-validate after changes

Paste into a notebook:
```python
from feature_engineering import build_features, clean_for_benchmarks
df = build_features(df_raw)

# Structural
assert df.shape[1] == 67  # update if you added columns
print(df.isna().sum().pipe(lambda s: s[s > 0]))

# Composite scores should be ~z-distributed
print(df[["hard_to_fill_score", "demand_intensity_score", "posting_quality_score"]]
      .describe().loc[["mean", "std", "min", "max"]])

# Within-category z-scores should be exactly mean 0, std 1 per category
print(df.groupby("category_1", observed=True)["salary_zscore_within_category"]
      .agg(["mean", "std"]).head())
```

### Tune the clean-benchmark filter

`clean_for_benchmarks(df, salary_upper_bound=50000)` exposes one knob. On this MCF dataset the upper bound is a no-op (top salaries are S$20k). Lower it to `19999` if you want to exclude ceiling-capped rows from benchmarks.

### Rebuild the feature parquet

```bash
python -c "from feature_engineering import build_features; import pandas as pd; \
  build_features(pd.read_parquet('raw.parquet')).to_parquet('data/mcf_features.parquet')"
```

This is the only data artifact the Streamlit app consumes. Re-run whenever the raw data refreshes.

---

## Tooling

- **Python**: 3.10+ required (uses `X | None` type hints in `feature_engineering.py`).
- **Linting**: none enforced; match existing style.
- **Tests**: smoke test in `feature_engineering.py` (`__main__` block). No pytest suite yet.
- **Notebooks**: notebooks are under coursework folders. The MCF analysis notebook is `step2_eda.ipynb` at repo root.

---

## Things explicitly *not* in scope

- **Skill tag extraction** from titles (Python, Java, AWS, etc.) — flagged in the dashboard spec as v2.
- **Geographic features** (Singapore districts from titles) — v2.
- **Live data refresh / scraping** — feed a Parquet file in; no integration with MCF API.
- **Multi-tenant auth** — single-user app.
- **PDF report export** — v2.

If you find yourself building these, check the dashboard spec section 12 first.

---

## When in doubt

- Architecture / pipeline questions → read `feature_engineering.py` docstrings.
- App-build questions → read `recruiter_dashboard_spec.md`.
- Business framing → section 1 of the spec.
- Data quirks → section "Data quirks to know" above.

Default to: simplicity over cleverness, prose explanations over jargon, direct support for the business objective ("identify promising roles and benchmark salaries").

## Logging of changes
- include changes to release.md in the commit message when you make changes to the codebase

